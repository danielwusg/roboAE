from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

from robot_auto_evolve.protocol.schema import StrictSchemaError, string

from .identity import ServiceIdentity


class ServiceClient(Protocol):
    def call(
        self,
        method: str,
        payload: Any,
        *,
        session_id: str,
        request_id: str | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class ServiceReplica:
    endpoint: str
    identity: ServiceIdentity
    client: ServiceClient

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", string(self.endpoint, "replica.endpoint"))
        if not isinstance(self.identity, ServiceIdentity):
            raise StrictSchemaError("replica.identity: expected ServiceIdentity")
        if not hasattr(self.client, "call"):
            raise StrictSchemaError("replica.client: expected call method")


class SessionCapacityError(RuntimeError):
    pass


class UnknownSessionError(RuntimeError):
    pass


class ReplicaScheduler:
    def __init__(
        self,
        replicas: tuple[ServiceReplica, ...] | list[ServiceReplica],
        *,
        max_sessions_per_replica: int = 1,
    ) -> None:
        replicas = tuple(replicas)
        if not replicas or any(not isinstance(replica, ServiceReplica) for replica in replicas):
            raise StrictSchemaError("scheduler.replicas: expected nonempty ServiceReplica sequence")
        if any(len(replica.identity.gpu_ids) != 1 for replica in replicas):
            raise StrictSchemaError("scheduler.replicas: each replica must use one GPU")
        if len({replica.identity.gpu_ids for replica in replicas}) != len(replicas):
            raise StrictSchemaError("scheduler.replicas: duplicate GPU assignment")
        if len({replica.identity.replica_id for replica in replicas}) != len(replicas):
            raise StrictSchemaError("scheduler.replicas: duplicate replica_id")
        if len({replica.endpoint for replica in replicas}) != len(replicas):
            raise StrictSchemaError("scheduler.replicas: duplicate endpoint")
        if any(not replicas[0].identity.same_model_as(replica.identity) for replica in replicas[1:]):
            raise StrictSchemaError("scheduler.replicas: model identities differ")
        if type(max_sessions_per_replica) is not int or max_sessions_per_replica < 1:
            raise StrictSchemaError("scheduler.max_sessions_per_replica: expected positive int")
        self.replicas = replicas
        self.replica_count = len(replicas)
        self.stateful = replicas[0].identity.stateful
        self.max_sessions_per_replica = max_sessions_per_replica
        self._condition = threading.Condition(threading.RLock())
        self._pinned: dict[str, int] = {}
        self._owners: list[set[str]] = [set() for _ in replicas]
        self._inflight = [0 for _ in replicas]
        self._session_locks: dict[str, threading.Lock] = {}
        self._replica_locks = [threading.Lock() for _ in replicas]
        self._next = 0

    def _remaining(self, deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def open_session(
        self,
        session_id: str,
        *,
        timeout: float | None = None,
        preferred_replica_id: str | None = None,
    ) -> ServiceReplica:
        session_id = string(session_id, "session_id")
        preferred = None
        if preferred_replica_id is not None:
            replica_id = string(preferred_replica_id, "preferred_replica_id")
            matches = [index for index, replica in enumerate(self.replicas) if replica.identity.replica_id == replica_id]
            if len(matches) != 1:
                raise StrictSchemaError("preferred_replica_id: unknown replica")
            preferred = matches[0]
        if timeout is not None and timeout < 0:
            raise StrictSchemaError("timeout: expected nonnegative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            if session_id in self._pinned:
                index = self._pinned[session_id]
                if preferred is not None and index != preferred:
                    raise StrictSchemaError("session is pinned to a different preferred replica")
                return self.replicas[index]
            if not self.stateful:
                index = self._select_stateless() if preferred is None else preferred
                self._pinned[session_id] = index
                self._session_locks[session_id] = threading.Lock()
                return self.replicas[index]
            while True:
                candidates = range(self.replica_count) if preferred is None else (preferred,)
                available = [index for index in candidates if len(self._owners[index]) < self.max_sessions_per_replica]
                if available:
                    index = min(
                        available,
                        key=lambda item: (
                            len(self._owners[item]),
                            self._inflight[item],
                            (item - self._next) % self.replica_count,
                        ),
                    )
                    self._next = (index + 1) % self.replica_count
                    self._owners[index].add(session_id)
                    self._pinned[session_id] = index
                    self._session_locks[session_id] = threading.Lock()
                    return self.replicas[index]
                remaining = self._remaining(deadline)
                if remaining is not None and remaining <= 0:
                    raise SessionCapacityError("no stateful replica available")
                self._condition.wait(remaining)

    def _select_stateless(self) -> int:
        return min(
            range(self.replica_count),
            key=lambda item: (self._inflight[item], (item - self._next) % self.replica_count),
        )

    def replica_for(self, session_id: str) -> ServiceReplica:
        session_id = string(session_id, "session_id")
        with self._condition:
            if session_id not in self._pinned:
                raise UnknownSessionError(session_id)
            return self.replicas[self._pinned[session_id]]

    def close_session(self, session_id: str) -> None:
        session_id = string(session_id, "session_id")
        with self._condition:
            if session_id not in self._pinned:
                raise UnknownSessionError(session_id)
            index = self._pinned[session_id]
            if self._inflight[index] != 0:
                raise RuntimeError("session has an in-flight call")
            del self._pinned[session_id]
            del self._session_locks[session_id]
            if self.stateful:
                if session_id not in self._owners[index]:
                    raise RuntimeError("scheduler ownership corrupted")
                self._owners[index].remove(session_id)
            self._condition.notify_all()

    @contextmanager
    def session(
        self,
        session_id: str,
        *,
        timeout: float | None = None,
        preferred_replica_id: str | None = None,
    ) -> Iterator[ServiceReplica]:
        replica = self.open_session(
            session_id,
            timeout=timeout,
            preferred_replica_id=preferred_replica_id,
        )
        try:
            yield replica
        finally:
            self.close_session(session_id)

    def call(
        self,
        session_id: str,
        method: str,
        payload: Any,
        *,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        session_id = string(session_id, "session_id")
        method = string(method, "method")
        self.open_session(session_id, timeout=timeout)
        with self._condition:
            index = self._pinned[session_id]
            lock = self._session_locks[session_id]
        with lock:
            with self._replica_locks[index]:
                with self._condition:
                    if self._pinned.get(session_id) != index:
                        raise UnknownSessionError(session_id)
                    self._inflight[index] += 1
                try:
                    return self.replicas[index].client.call(
                        method,
                        payload,
                        session_id=session_id,
                        request_id=request_id,
                    )
                finally:
                    with self._condition:
                        self._inflight[index] -= 1
                        self._condition.notify_all()

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_auto_evolve.protocol.schema import StrictSchemaError, boolean, fields, integer, mapping, number, sha256, string

from .routes import PolicyRoute, route


@dataclass(frozen=True)
class PolicyServiceConfig:
    path: Path
    value: dict[str, Any]
    route: PolicyRoute
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "PolicyServiceConfig":
        source = Path(path).resolve()
        if source.suffix.lower() != ".json":
            raise StrictSchemaError("policy config must be JSON")
        value = json.loads(source.read_text(encoding="utf-8"))
        common = {"schema_version", "route", "implementation", "checkpoint"}
        route_name = string(mapping(value, "policy_config").get("route"), "policy_config.route")
        selected = route(route_name)
        backend_fields = {
            "xvla": {"torch_dtype", "denoise_steps", "action_horizon", "trust_remote_code"},
            "openvla": {"reference_file", "reference_file_commit", "reference_file_sha256", "attn_implementation", "unnorm_key", "image_size", "image_resize", "prompt_format", "sticky_gripper_steps", "sticky_gripper_preserve_previous", "action_horizon", "execution_count", "deployment", "torch_dtype", "trust_remote_code", "status"},
            "pi05": {"upstream_config", "action_horizon", "execution_count", "deployment", "torch_dtype", "compile_mode", "status"},
            "smolvla": {"base_model", "action_horizon", "execution_count", "runtime_state_width", "deployment", "torch_dtype", "status"},
            "rlinf_pi05": {"runtime_implementation", "upstream_config", "action_horizon", "execution_count", "denoise_steps", "deployment", "torch_dtype", "compile_mode", "norm_asset_id", "use_quantile_norm", "status"},
            "molmoact2": {"inference_action_mode", "norm_tag", "num_steps", "action_horizon", "execution_count", "deployment", "torch_dtype", "enable_depth_reasoning", "enable_adaptive_depth", "enable_cuda_graph", "normalize_language", "status"},
            "molmoact2_droid": {"inference_action_mode", "norm_tag", "num_steps", "action_horizon", "execution_count", "deployment", "torch_dtype", "enable_cuda_graph", "normalize_language", "camera_layout", "status"},
            "rldx": {"action_horizon", "execution_count", "deployment", "torch_dtype", "status"},
            "molmobot": {"tokenizer", "action_horizon", "execution_count", "deployment", "torch_dtype", "num_steps", "states_mode", "observation_steps", "observation_step_delta", "max_joint_delta", "status"},
            "openpi_droid_jointpos": {"upstream_config", "checkpoint_path", "artifact_manifest_path", "paligemma_tokenizer_path", "paligemma_tokenizer_artifact_manifest_path", "paligemma_tokenizer_sha256", "action_dim", "action_horizon", "execution_count", "deployment", "compute_dtype", "sampling", "camera_layout", "status"},
            "dreamzero": {"action_horizon", "execution_count", "deployment", "status"},
        }
        backend_optional_fields = {
            "openpi_droid_jointpos": {"num_steps", "max_decoding_steps", "fast_tokenizer"},
        }
        obj = fields(
            value,
            common | backend_fields[selected.backend],
            optional=backend_optional_fields.get(selected.backend, frozenset()),
            path="policy_config",
        )
        if integer(obj.get("schema_version"), "policy_config.schema_version") != 1:
            raise StrictSchemaError("policy_config.schema_version must be 1")
        implementation = fields(
            obj["implementation"],
            {"repository", "commit"},
            path="policy_config.implementation",
        )
        checkpoint = fields(obj["checkpoint"], {"id", "revision"}, path="policy_config.checkpoint")
        pins = {
            "policy_config.implementation.repository": (implementation["repository"], selected.repository),
            "policy_config.implementation.commit": (implementation["commit"], selected.source_commit),
            "policy_config.checkpoint.id": (checkpoint["id"], selected.model_id),
            "policy_config.checkpoint.revision": (checkpoint["revision"], selected.revision),
        }
        for name, (actual, expected) in pins.items():
            if string(actual, name) != expected:
                raise StrictSchemaError(f"{name}: route pin mismatch")
        if selected.backend == "xvla":
            if obj["torch_dtype"] != "float32" or integer(obj["denoise_steps"], "policy_config.denoise_steps") != 10:
                raise StrictSchemaError("policy_config: X-VLA requires float32 and 10 denoise steps")
            if integer(obj["action_horizon"], "policy_config.action_horizon") != selected.action_horizon:
                raise StrictSchemaError("policy_config.action_horizon: route mismatch")
            if not boolean(obj["trust_remote_code"], "policy_config.trust_remote_code"):
                raise StrictSchemaError("policy_config.trust_remote_code: expected true")
        else:
            if obj["status"] != selected.status:
                raise StrictSchemaError("policy_config.status: route mismatch")
            if obj["deployment"] not in {"replicated", "tensor_parallel"}:
                raise StrictSchemaError("policy_config.deployment: unsupported value")
            if "action_horizon" in obj and integer(obj["action_horizon"], "policy_config.action_horizon") != selected.action_horizon:
                raise StrictSchemaError("policy_config.action_horizon: route mismatch")
            expected_counts = {"openvla": 1, "smolvla": 8, "rlinf_pi05": 5, "molmoact2": 10, "molmoact2_droid": 15, "rldx": 8, "molmobot": 8, "dreamzero": 24}
            if selected.backend in expected_counts and integer(obj["execution_count"], "policy_config.execution_count") != expected_counts[selected.backend]:
                raise StrictSchemaError("policy_config.execution_count: route mismatch")
        if selected.backend == "pi05":
            if (
                obj["upstream_config"] != "pi05_libero"
                or obj["deployment"] != "replicated"
                or obj["torch_dtype"] != "bfloat16"
                or obj["compile_mode"] != "max-autotune-no-cudagraphs"
            ):
                raise StrictSchemaError("policy_config: invalid pi0.5 LIBERO settings")
            if integer(obj["action_horizon"], "policy_config.action_horizon") != 50:
                raise StrictSchemaError("policy_config.action_horizon: route mismatch")
            if integer(obj["execution_count"], "policy_config.execution_count") != 10:
                raise StrictSchemaError("policy_config.execution_count: route mismatch")
        if selected.backend == "openvla":
            if (
                obj["reference_file"] != "simpler_env/policies/openvla/openvla_model.py"
                or obj["reference_file_commit"] != "06b0cf23d3eb7f572c888993a042037336d1a52c"
                or obj["reference_file_sha256"] != "74da205be0de0c86b4219d99393dc92fbf0e92fc2190bd0144ae4ce6c30cdc7b"
                or obj["attn_implementation"] != "flash_attention_2"
                or obj["unnorm_key"] != "fractal20220817_data"
                or integer(obj["image_size"], "policy_config.image_size") != 224
                or obj["image_resize"] != "opencv_inter_area"
                or obj["prompt_format"] != "raw_task_description"
                or integer(obj["sticky_gripper_steps"], "policy_config.sticky_gripper_steps") != 15
                or not boolean(obj["sticky_gripper_preserve_previous"], "policy_config.sticky_gripper_preserve_previous")
                or obj["deployment"] != "replicated"
                or obj["torch_dtype"] != "bfloat16"
                or not boolean(obj["trust_remote_code"], "policy_config.trust_remote_code")
            ):
                raise StrictSchemaError("policy_config: invalid OpenVLA SimplerEnv Google settings")
        if selected.backend == "smolvla":
            base = fields(obj["base_model"], {"id", "revision"}, path="policy_config.base_model")
            if (
                base["id"] != "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
                or base["revision"] != "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
                or obj["deployment"] != "replicated"
                or obj["torch_dtype"] != "bfloat16"
                or integer(obj["runtime_state_width"], "policy_config.runtime_state_width") != 8
            ):
                raise StrictSchemaError("policy_config: invalid SmolVLA RoboCerebra settings")
        if selected.backend == "rlinf_pi05":
            runtime = fields(
                obj["runtime_implementation"],
                {"repository", "commit"},
                path="policy_config.runtime_implementation",
            )
            if (
                runtime["repository"] != "https://github.com/RLinf/openpi.git"
                or runtime["commit"] != "c5dc4b9296a1a4739bf52828f28a579f12dce763"
                or obj["upstream_config"] != "pi05_libero"
                or integer(obj["denoise_steps"], "policy_config.denoise_steps") != 10
                or obj["deployment"] != "replicated"
                or obj["torch_dtype"] != "bfloat16"
                or obj["compile_mode"] != "max-autotune-no-cudagraphs"
                or obj["norm_asset_id"] != "physical-intelligence/libero"
                or not boolean(obj["use_quantile_norm"], "policy_config.use_quantile_norm")
            ):
                raise StrictSchemaError("policy_config: invalid RLinf pi0.5 LIBERO-Pro settings")
        if selected.backend == "molmoact2":
            expected_depth = selected.name == "molmoact2_think_libero"
            if (
                obj["inference_action_mode"] != "continuous"
                or obj["norm_tag"] != "libero"
                or integer(obj["num_steps"], "policy_config.num_steps") != 10
                or obj["deployment"] != "replicated"
                or obj["torch_dtype"] != "float32"
                or boolean(obj["enable_depth_reasoning"], "policy_config.enable_depth_reasoning") != expected_depth
                or boolean(obj["enable_adaptive_depth"], "policy_config.enable_adaptive_depth") != expected_depth
                or not boolean(obj["enable_cuda_graph"], "policy_config.enable_cuda_graph")
                or not boolean(obj["normalize_language"], "policy_config.normalize_language")
            ):
                raise StrictSchemaError("policy_config: invalid MolmoAct2 LIBERO settings")
        if selected.backend == "molmoact2_droid":
            if (
                obj["inference_action_mode"] != "continuous"
                or obj["norm_tag"] != "franka_droid"
                or integer(obj["num_steps"], "policy_config.num_steps") != 10
                or obj["deployment"] != "replicated"
                or obj["torch_dtype"] != "float32"
                or not boolean(obj["enable_cuda_graph"], "policy_config.enable_cuda_graph")
                or not boolean(obj["normalize_language"], "policy_config.normalize_language")
                or obj["camera_layout"] != "external_external_wrist"
            ):
                raise StrictSchemaError("policy_config: invalid MolmoAct2 DROID settings")
        if selected.backend == "rldx" and (obj["deployment"] != "replicated" or obj["torch_dtype"] != "bfloat16"):
            raise StrictSchemaError("policy_config: invalid RLDX deployment")
        if selected.backend == "molmobot":
            tokenizer = fields(obj["tokenizer"], {"id", "revision"}, path="policy_config.tokenizer")
            if (
                obj["deployment"] != "replicated"
                or obj["torch_dtype"] != "bfloat16"
                or tokenizer["id"] != "Qwen/Qwen3-4B-Instruct-2507"
                or tokenizer["revision"] != "f50518eb58dfc750271b273fc113bdfc16ec2280"
                or integer(obj["num_steps"], "policy_config.num_steps") != 10
                or obj["states_mode"] != "cross_attn"
                or integer(obj["observation_steps"], "policy_config.observation_steps") != 2
                or integer(obj["observation_step_delta"], "policy_config.observation_step_delta") != 8
                or number(obj["max_joint_delta"], "policy_config.max_joint_delta") != 0.2
            ):
                raise StrictSchemaError("policy_config: invalid MolmoBot deployment")
        if selected.backend == "openpi_droid_jointpos":
            pi05 = selected.name == "openpi_pi05_droid_jointpos"
            expected_name = "pi05_droid_jointpos" if pi05 else "pi0_fast_droid_jointpos"
            expected_checkpoint = f"test_runs/_shared/openpi_robolab_checkpoints/{expected_name}/checkpoint"
            expected_artifact = f"test_runs/_shared/openpi_robolab_checkpoints/{expected_name}/artifact.json"
            if (
                obj["upstream_config"] != expected_name
                or obj["checkpoint_path"] != expected_checkpoint
                or obj["artifact_manifest_path"] != expected_artifact
                or obj["paligemma_tokenizer_path"] != "test_runs/_shared/openpi_robolab_checkpoints/tokenizers/paligemma_tokenizer.model"
                or obj["paligemma_tokenizer_artifact_manifest_path"] != "test_runs/_shared/openpi_robolab_checkpoints/tokenizers/paligemma_tokenizer.artifact.json"
                or sha256(obj["paligemma_tokenizer_sha256"], "policy_config.paligemma_tokenizer_sha256") != "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
                or integer(obj["action_dim"], "policy_config.action_dim") != (32 if pi05 else 8)
                or integer(obj["execution_count"], "policy_config.execution_count") != selected.action_horizon
                or obj["deployment"] != "replicated"
                or obj["compute_dtype"] != "bfloat16"
                or obj["sampling"] != ("explicit_seeded_noise" if pi05 else "greedy")
                or obj["camera_layout"] != "external_wrist"
            ):
                raise StrictSchemaError("policy_config: invalid OpenPI DROID joint-position settings")
            if pi05:
                if set(obj) & {"max_decoding_steps", "fast_tokenizer"} or integer(obj.get("num_steps"), "policy_config.num_steps") != 10:
                    raise StrictSchemaError("policy_config: invalid OpenPI pi0.5 sampling settings")
            else:
                if "num_steps" in obj or integer(obj.get("max_decoding_steps"), "policy_config.max_decoding_steps") != 256:
                    raise StrictSchemaError("policy_config: invalid OpenPI pi0-FAST sampling settings")
                tokenizer = fields(obj.get("fast_tokenizer"), {"id", "revision", "expected_files"}, path="policy_config.fast_tokenizer")
                if tokenizer["id"] != "physical-intelligence/fast" or tokenizer["revision"] != "ec4d7aa71691cac0b8bed6942be45684db2110f4":
                    raise StrictSchemaError("policy_config: invalid FAST tokenizer identity")
                expected_files = {
                    ".gitattributes": (1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"),
                    "README.md": (3243, "eeb548bbc962193940ee078225752c2cd8df91fda904467fc2206f641743038f"),
                    "processing_action_tokenizer.py": (6145, "6f021ca1f4c1b194ab6fa399d80baf3d642eadb17efb8f73301e4ac401522c20"),
                    "processor_config.json": (253, "f40cfbb1020858fe1d48c0f946b0c1315a90d6e84aa82710036f24f4c167706a"),
                    "special_tokens_map.json": (3, "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"),
                    "tokenizer.json": (686974, "6507dd709287fd018882120c0071787f1f62bad9f180f1e8c5235bda1b71fa78"),
                    "tokenizer_config.json": (322, "b4030e2a13a0dea22e99d54c086fb320c71e66ad034ac4eba4301a0a27d5e5cd"),
                }
                files = mapping(tokenizer["expected_files"], "policy_config.fast_tokenizer.expected_files")
                if set(files) != set(expected_files):
                    raise StrictSchemaError("policy_config: invalid FAST tokenizer file set")
                for name, (size, digest) in expected_files.items():
                    item = fields(files[name], {"size_bytes", "sha256"}, path=f"policy_config.fast_tokenizer.expected_files.{name}")
                    if integer(item["size_bytes"], f"policy_config.fast_tokenizer.expected_files.{name}.size_bytes") != size or sha256(item["sha256"], f"policy_config.fast_tokenizer.expected_files.{name}.sha256") != digest:
                        raise StrictSchemaError(f"policy_config: invalid FAST tokenizer file {name}")
        if selected.backend == "dreamzero" and obj["deployment"] != "tensor_parallel":
            raise StrictSchemaError("policy_config: invalid DreamZero deployment")
        checked = mapping(value, "policy_config")
        canonical = json.dumps(checked, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return cls(source, dict(checked), selected, hashlib.sha256(canonical).hexdigest())

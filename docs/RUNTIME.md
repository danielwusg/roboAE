# Runtime artifacts

The clean repository reuses the existing runtime artifacts. It does not copy model weights, Conda environments, simulators, datasets, assets, or upstream source repositories.

The site path map points to:

- environments: `/nlp/scr/shgwu/roboDev/newcode/conda/envs`
- upstream sources: `/nlp/scr/shgwu/roboDev/newcode/external`
- shared Hugging Face cache: `/nlp/scr/shgwu/.cache/huggingface/hub`
- existing locked assets and checkpoints under `/nlp/scr/shgwu/roboDev/newcode/test_runs`

Run the read-only preflight from the repository root:

```bash
./launch/verify_runtime.sh
```

The report keeps two scopes separate. `verified` means the preflight checked the committed identity: Conda history and Python version for 21 environments, clean-package import origin for every environment, Git commit and clean worktree for pinned sources, hashes for the three existing runtime lock or manifest files, and the full derived X-VLA SimplerEnv tree. `declared_only` means the path exists but this portable lock does not identify all of its content: the shared Hugging Face and compilation caches, LIBERO dataset directories, LIBERO-Pro assets and dataset directory, and extracted RoboCerebra assets. Route startup performs its own checkpoint and asset checks where an adapter provides them; the runtime preflight does not relabel those path-only items as verified.

The reused environments contain editable bindings to the development source tree. Every clean process therefore prepends `/nlp/scr/shgwu/roboAE/src`, disables user site packages, and verifies that `robot_auto_evolve` was imported only from the clean repository. Intended editable bindings to pinned upstream policy and simulator repositories remain allowed after commit and clean-tree verification.

Run-owned process state is rejected unless it is below `runs/<run-id>/runtime`. Credentials remain outside both repositories. Existing shared compilation and model caches are performance aids and are never treated as experiment identity.

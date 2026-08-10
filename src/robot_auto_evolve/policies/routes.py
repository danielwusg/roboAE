from __future__ import annotations

from dataclasses import dataclass


class UnavailablePolicyRoute(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicyRoute:
    name: str
    model_id: str | None
    revision: str | None
    source_commit: str | None
    backend: str | None
    status: str
    repository: str | None = None
    native_action_width: int | None = None
    action_horizon: int | None = None
    blocker: str | None = None

    def require_available(self) -> "PolicyRoute":
        if self.status != "backend_implemented":
            raise UnavailablePolicyRoute(f"{self.name}: {self.blocker}")
        return self


ROUTES = {
    "xvla_libero": PolicyRoute("xvla_libero", "2toINF/X-VLA-Libero", "129e71460678b7236cee6fc9707f09d9fa0c3590", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "xvla", "backend_implemented", "https://github.com/2toinf/X-VLA.git", 20, 30),
    "xvla_calvin": PolicyRoute("xvla_calvin", "2toINF/X-VLA-Calvin-ABC_D", "d76710ee314ee1fa8506f421664c989b40bae415", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "xvla", "backend_implemented", "https://github.com/2toinf/X-VLA.git", 20, 30),
    "xvla_simpler_widowx": PolicyRoute("xvla_simpler_widowx", "2toINF/X-VLA-WidowX", "8d7ea1aaa948665d44129a3ff488629b955fc0f9", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "xvla", "backend_implemented", "https://github.com/2toinf/X-VLA.git", 20, 30),
    "xvla_simpler_google_va": PolicyRoute("xvla_simpler_google_va", "2toINF/X-VLA-Google-Robot", "afaad7ac52e483629e688f0c9c681cc58472d130", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "xvla", "backend_implemented", "https://github.com/2toinf/X-VLA.git", 20, 30),
    "xvla_simpler_google_vm": PolicyRoute("xvla_simpler_google_vm", "2toINF/X-VLA-Google-Robot", "afaad7ac52e483629e688f0c9c681cc58472d130", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "xvla", "backend_implemented", "https://github.com/2toinf/X-VLA.git", 20, 30),
    "xvla_pt_simpler_google_va": PolicyRoute("xvla_pt_simpler_google_va", "2toINF/X-VLA-Pt", "c1c4a64a7e03ac5b95c468bf1578f3d03651b53b", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "xvla", "backend_implemented", "https://github.com/2toinf/X-VLA.git", 20, 30),
    "xvla_pt_simpler_google_vm": PolicyRoute("xvla_pt_simpler_google_vm", "2toINF/X-VLA-Pt", "c1c4a64a7e03ac5b95c468bf1578f3d03651b53b", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "xvla", "backend_implemented", "https://github.com/2toinf/X-VLA.git", 20, 30),
    "xvla_pt_libero": PolicyRoute("xvla_pt_libero", "2toINF/X-VLA-Pt", "c1c4a64a7e03ac5b95c468bf1578f3d03651b53b", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "xvla", "backend_implemented", "https://github.com/2toinf/X-VLA.git", 20, 30),
    "xvla_robocerebra": PolicyRoute("xvla_robocerebra", "2toINF/X-VLA-Libero", "129e71460678b7236cee6fc9707f09d9fa0c3590", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "xvla", "backend_implemented", "https://github.com/2toinf/X-VLA.git", 20, 30),
    "xvla_pt_robocerebra": PolicyRoute("xvla_pt_robocerebra", "2toINF/X-VLA-Pt", "c1c4a64a7e03ac5b95c468bf1578f3d03651b53b", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "xvla", "backend_implemented", "https://github.com/2toinf/X-VLA.git", 20, 30),
    "xvla_pt_simpler_widowx": PolicyRoute("xvla_pt_simpler_widowx", "2toINF/X-VLA-Pt", "c1c4a64a7e03ac5b95c468bf1578f3d03651b53b", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "xvla", "backend_implemented", "https://github.com/2toinf/X-VLA.git", 20, 30),
    "xvla_pt_vlabench": PolicyRoute("xvla_pt_vlabench", "2toINF/X-VLA-Pt", "c1c4a64a7e03ac5b95c468bf1578f3d03651b53b", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "xvla", "backend_implemented", "https://github.com/2toinf/X-VLA.git", 20, 30),
    "openvla_pt_simpler_google_va": PolicyRoute("openvla_pt_simpler_google_va", "openvla/openvla-7b", "47a0ec7fc4ec123775a391911046cf33cf9ed83f", "ccfe3809766839a2fcfb7a3d3c9abff585189188", "openvla", "backend_implemented", "https://github.com/DelinQu/SimplerEnv-OpenVLA.git", 7, 1),
    "openvla_pt_simpler_google_vm": PolicyRoute("openvla_pt_simpler_google_vm", "openvla/openvla-7b", "47a0ec7fc4ec123775a391911046cf33cf9ed83f", "ccfe3809766839a2fcfb7a3d3c9abff585189188", "openvla", "backend_implemented", "https://github.com/DelinQu/SimplerEnv-OpenVLA.git", 7, 1),
    "openvla_libero_spatial": PolicyRoute("openvla_libero_spatial", "openvla/openvla-7b-finetuned-libero-spatial", "962318cec55ac10993ff0f5f43eda9a270b4c873", "ccfe3809766839a2fcfb7a3d3c9abff585189188", "openvla", "backend_implemented", "https://github.com/DelinQu/SimplerEnv-OpenVLA.git", 7, 1),
    "openvla_libero_object": PolicyRoute("openvla_libero_object", "openvla/openvla-7b-finetuned-libero-object", "287d6cfdf12d07b1449505f66d9bf3550257e9b3", "ccfe3809766839a2fcfb7a3d3c9abff585189188", "openvla", "backend_implemented", "https://github.com/DelinQu/SimplerEnv-OpenVLA.git", 7, 1),
    "openvla_libero_goal": PolicyRoute("openvla_libero_goal", "openvla/openvla-7b-finetuned-libero-goal", "fa5ae1e7509348889295bba8e08621d8b55e9baf", "ccfe3809766839a2fcfb7a3d3c9abff585189188", "openvla", "backend_implemented", "https://github.com/DelinQu/SimplerEnv-OpenVLA.git", 7, 1),
    "openvla_libero_10": PolicyRoute("openvla_libero_10", "openvla/openvla-7b-finetuned-libero-10", "80970322773f81baa2e22fe495d0487b93a05cfa", "ccfe3809766839a2fcfb7a3d3c9abff585189188", "openvla", "backend_implemented", "https://github.com/DelinQu/SimplerEnv-OpenVLA.git", 7, 1),
    "openvla_robocerebra": PolicyRoute("openvla_robocerebra", "openvla/openvla-7b-finetuned-libero-10", "80970322773f81baa2e22fe495d0487b93a05cfa", "ccfe3809766839a2fcfb7a3d3c9abff585189188", "openvla", "backend_implemented", "https://github.com/DelinQu/SimplerEnv-OpenVLA.git", 7, 1),
    "xvla_robotwin2": PolicyRoute("xvla_robotwin2", "2toINF/X-VLA-RoboTwin2", "a157c580cfe6f9f445614490f3bec1b2f9ef9f18", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "xvla", "backend_implemented", "https://github.com/2toinf/X-VLA.git", 20, 30),
    "xvla_vlabench": PolicyRoute("xvla_vlabench", "2toINF/X-VLA-VLABench", "0995f2f51c9f2e29d78f20080948d25ce7e28d88", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "xvla", "backend_implemented", "https://github.com/2toinf/X-VLA.git", 20, 30),
    "pi05_libero": PolicyRoute("pi05_libero", "lerobot/pi05_libero_finetuned", "dbf8a3f794a9c4297b44f40b752712f50073d945", "8fff0fde7c79f23a93d845d1a50e985de01f8b8a", "pi05", "backend_implemented", "https://github.com/huggingface/lerobot.git", 7, 50),
    "smolvla_robocerebra": PolicyRoute("smolvla_robocerebra", "lerobot/smolvla_robocerebra", "7ff416240ff73bda10a2b5dbd4245f72eaa959d0", "8fff0fde7c79f23a93d845d1a50e985de01f8b8a", "smolvla", "backend_implemented", "https://github.com/huggingface/lerobot.git", 7, 50),
    "smolvla_libero_pro": PolicyRoute("smolvla_libero_pro", "lerobot/smolvla_robocerebra", "7ff416240ff73bda10a2b5dbd4245f72eaa959d0", "8fff0fde7c79f23a93d845d1a50e985de01f8b8a", "smolvla", "backend_implemented", "https://github.com/huggingface/lerobot.git", 7, 50),
    "rlinf_pi05_libero": PolicyRoute("rlinf_pi05_libero", "RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT", "6222623f635769bfc73c9472e29fab9b7fd8e027", "c5ca51cc21c007a41d287159f9e1b14e0200000e", "rlinf_pi05", "backend_implemented", "https://github.com/RLinf/RLinf.git", 7, 10),
    "rlinf_pi05_libero_pro": PolicyRoute("rlinf_pi05_libero_pro", "RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT", "6222623f635769bfc73c9472e29fab9b7fd8e027", "c5ca51cc21c007a41d287159f9e1b14e0200000e", "rlinf_pi05", "backend_implemented", "https://github.com/RLinf/RLinf.git", 7, 10),
    "molmoact2_libero": PolicyRoute("molmoact2_libero", "allenai/MolmoAct2-LIBERO", "0d24a92bd1faf321ef497c3bbd5681af97c65aa2", "28b4f721f931aab554cb176412223f098023705f", "molmoact2", "backend_implemented", "https://github.com/allenai/lerobot.git", 7, 10),
    "molmoact2_think_libero": PolicyRoute("molmoact2_think_libero", "allenai/MolmoAct2-Think-LIBERO", "593d25fcd3150e38eb05812fc3f9adb02927ec83", "28b4f721f931aab554cb176412223f098023705f", "molmoact2", "backend_implemented", "https://github.com/allenai/lerobot.git", 7, 10),
    "molmoact2_droid": PolicyRoute("molmoact2_droid", "allenai/MolmoAct2-DROID", "d8c1abd8a27d8e859455bbe514df2bcc617db0fb", "c2282820f9b188b60e66ea1636b3efd81c45cbb4", "molmoact2_droid", "backend_implemented", "https://github.com/allenai/molmoact2.git", 8, 15),
    "molmoact2_robocerebra": PolicyRoute("molmoact2_robocerebra", "allenai/MolmoAct2-LIBERO", "0d24a92bd1faf321ef497c3bbd5681af97c65aa2", "28b4f721f931aab554cb176412223f098023705f", "molmoact2", "backend_implemented", "https://github.com/allenai/lerobot.git", 7, 10),
    "molmoact2_think_robocerebra": PolicyRoute("molmoact2_think_robocerebra", "allenai/MolmoAct2-Think-LIBERO", "593d25fcd3150e38eb05812fc3f9adb02927ec83", "28b4f721f931aab554cb176412223f098023705f", "molmoact2", "backend_implemented", "https://github.com/allenai/lerobot.git", 7, 10),
    "pi05_robocerebra": PolicyRoute("pi05_robocerebra", "lerobot/pi05_libero_finetuned", "dbf8a3f794a9c4297b44f40b752712f50073d945", "8fff0fde7c79f23a93d845d1a50e985de01f8b8a", "pi05", "backend_implemented", "https://github.com/huggingface/lerobot.git", 7, 50),
    "rlinf_pi05_robocerebra": PolicyRoute("rlinf_pi05_robocerebra", "RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT", "6222623f635769bfc73c9472e29fab9b7fd8e027", "c5ca51cc21c007a41d287159f9e1b14e0200000e", "rlinf_pi05", "backend_implemented", "https://github.com/RLinf/RLinf.git", 7, 10),
    "rldx_robocasa365": PolicyRoute("rldx_robocasa365", "RLWRLD/RLDX-1-FT-RC365", "587e9ecdcc5e7184fcc17f58713908edff5af041", "ebbfb4f6214bb38de07da1a70f597201feceb6da", "rldx", "backend_implemented", "https://github.com/RLWRLD/RLDX-1.git", 12, 16),
    "molmobot_droid": PolicyRoute("molmobot_droid", "allenai/MolmoBot-DROID", "cbe6ec358958d07ddfb20d3aa54e560e9e1b18c9", "33c0ca77bf6062a23d60ffd4a6859334c4a46d30", "molmobot", "backend_implemented", "https://github.com/allenai/MolmoBot.git", 8, 16),
    "openpi_pi05_droid_jointpos": PolicyRoute("openpi_pi05_droid_jointpos", "gs://openpi-assets-simeval/pi05_droid_jointpos", "a07eac0836e88571bc718a649c3b9eb805a9e77e26c23dd24802a8d6b44027d0", "aa6420561529593114160d05e5ad155792b272f3", "openpi_droid_jointpos", "backend_implemented", "https://github.com/xuningy/openpi.git", 8, 15),
    "openpi_pi0_fast_droid_jointpos": PolicyRoute("openpi_pi0_fast_droid_jointpos", "gs://openpi-assets-simeval/pi0_fast_droid_jointpos", "4fc66bc5bd822e8e8d242600368ebec4f2807320925a69ee6f8d2a83f9bab48a", "aa6420561529593114160d05e5ad155792b272f3", "openpi_droid_jointpos", "backend_implemented", "https://github.com/xuningy/openpi.git", 8, 10),
    "dreamzero_droid": PolicyRoute("dreamzero_droid", "GEAR-Dreams/DreamZero-DROID", "96ad344138c66e82536422432ad742f015784942", "ab790c198fbce33503358efbbd4187ce9a89adf3", "dreamzero", "pinned_component_only", "https://github.com/dreamzero0/dreamzero.git", 8, 24, "Local policy backend and RoboLab worker are not integrated."),
    "lingbot_vla_v2_robotwin": PolicyRoute("lingbot_vla_v2_robotwin", "robbyant/lingbot-vla-v2-6b", "11c703bf6a5c1f45b3b69168482da11fdbba53d7", "69729b4ef24c63ec25e750915491635f4753be1d", None, "blocked", blocker="No public RoboTwin-post-trained checkpoint."),
    "harness_vla_exact": PolicyRoute("harness_vla_exact", None, None, None, None, "blocked", blocker="No public agent code, memory, or exact policy services."),
    "voloagent_exact": PolicyRoute("voloagent_exact", None, None, None, None, "blocked", blocker="No public agent code, RoboVoLo package, or complete checkpoint set."),
    "xvla_navsim": PolicyRoute("xvla_navsim", None, None, "6bc2513f5f1cbec715cc668b414392a6cae5c671", None, "blocked", blocker="No public NAVSIM checkpoint in the X-VLA collection."),
}


def route(name: str) -> PolicyRoute:
    try:
        return ROUTES[name]
    except KeyError as exc:
        raise UnavailablePolicyRoute(f"unknown policy route {name!r}") from exc

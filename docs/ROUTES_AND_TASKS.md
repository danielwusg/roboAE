# Route inventory

This inventory is generated from the strict route specifications. Task flags use the exact IDs in the tables. With no task flags, a ready full-benchmark wrapper evaluates every listed task unit on every unchanged standard row at every candidate. A held-out transfer task is never included in adaptive evidence and is evaluated only as baseline versus frozen scaffold after finalization.

Use `--task-preset related` for the pinned related-task study. Arbitrary disjoint evolve and transfer task lists remain mechanically allowed for new audited studies, but they do not by themselves support a related-task transfer claim.

A LIBERO suite wrapper covers one 10-task suite, and a LIBERO-Pro wrapper covers one 10-task cell. The group launchers are convenience batches of independent studies; because their members may freeze different scaffolds, their averages are not single-agent four-suite or eight-cell headlines.

## Slice convenience batches

| Group | Launcher | Members | Metric | Scope |
|---|---|---:|---|---|
| `xvla_libero_suite_slice_batch` | `launch/groups/xvla_libero_suite_slice_batch.sh` | 4 | not available | This starts four independent studies that may freeze different scaffolds. Their average is not a single-agent four-suite headline. |
| `lerobot_pi05_libero_suite_slice_batch` | `launch/groups/lerobot_pi05_libero_suite_slice_batch.sh` | 4 | not available | This starts four independent studies that may freeze different scaffolds. Their average is not a single-agent four-suite headline. |
| `molmoact2_base_libero_suite_slice_batch` | `launch/groups/molmoact2_base_libero_suite_slice_batch.sh` | 4 | not available | This starts four independent studies that may freeze different scaffolds. Their average is not a single-agent four-suite headline. |
| `molmoact2_think_libero_suite_slice_batch` | `launch/groups/molmoact2_think_libero_suite_slice_batch.sh` | 4 | not available | This starts four independent studies that may freeze different scaffolds. Their average is not a single-agent four-suite headline. |
| `rlinf_pi05_libero_suite_slice_batch` | `launch/groups/rlinf_pi05_libero_suite_slice_batch.sh` | 4 | not available | This starts four independent studies that may freeze different scaffolds. Their average is not a single-agent four-suite headline. |
| `rlinf_pi05_libero_pro_cell_slice_batch` | `launch/groups/rlinf_pi05_libero_pro_cell_slice_batch.sh` | 8 | not available | This starts eight independent studies that may freeze different scaffolds. Their average is not a single-agent eight-cell headline. |

## `molmoact2_libero`

- Route: MolmoAct2 Base + complete four-suite LIBERO
- Study role: canonical full benchmark
- Launcher: `launch/routes/libero/molmoact2_base_four_suite.sh`
- Profile: `configs/molmoact2_libero.json` (`4c5063b381b5d48ff82865bfef73745fb97221797a1f313f00e010d4aa80c1c6`)
- Profile set: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [allenai/MolmoAct2-LIBERO](https://huggingface.co/allenai/MolmoAct2-LIBERO/tree/0d24a92bd1faf321ef497c3bbd5681af97c65aa2) at `0d24a92bd1faf321ef497c3bbd5681af97c65aa2`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `molmoact2_libero_10_canonical_50_per_task_v1`, `molmoact2_libero_goal_canonical_50_per_task_v1`, `molmoact2_libero_object_canonical_50_per_task_v1`, `molmoact2_libero_spatial_canonical_50_per_task_v1`
- Standard route rows: 2000
- Comparability: This executes the complete standard four-suite plan with one shared evolving scaffold. The agent adds frozen tools, so it is comparable as an agent result, not as the raw policy baseline.
- Route benchmark plan: `manifests/benchmarks/molmoact2_libero_standard.json` (`965d82f2a695e567a3bb06946b3428769c4a6aaea12790b5c9048ad0eaf51527`)
- Exact standard source: `manifests/benchmarks/molmoact2_libero_standard.json` (`965d82f2a695e567a3bb06946b3428769c4a6aaea12790b5c9048ad0eaf51527`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`, `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`, `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`, `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`, `open_the_middle_drawer_of_the_cabinet`, `pick_up_the_alphabet_soup_and_place_it_in_the_basket`, `pick_up_the_bbq_sauce_and_place_it_in_the_basket`, `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_cream_cheese_and_place_it_in_the_basket`, `pick_up_the_ketchup_and_place_it_in_the_basket`, `pick_up_the_salad_dressing_and_place_it_in_the_basket`, `put_the_bowl_on_the_stove`, `put_the_bowl_on_top_of_the_cabinet`, `put_the_wine_bottle_on_top_of_the_cabinet`
- Preset held-out tasks: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`, `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`, `open_the_top_drawer_and_put_the_bowl_inside`, `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_butter_and_place_it_in_the_basket`, `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, `pick_up_the_milk_and_place_it_in_the_basket`, `pick_up_the_orange_juice_and_place_it_in_the_basket`, `pick_up_the_tomato_sauce_and_place_it_in_the_basket`, `put_the_bowl_on_the_plate`, `put_the_wine_bottle_on_the_rack`
- Preset sources: `manifests/episodes/molmoact2_libero_long_transfer.json` (`f417482fcab976342f907ae436440dc520fd613a632adc5f1940f2bc3ed8841b`), `manifests/episodes/molmoact2_libero_goal_transfer.json` (`aacbb4bf7c5008bc05187534b570da02b0283bc8b3acf9ad2b6b636f564f705d`), `manifests/episodes/molmoact2_libero_object_transfer.json` (`89310c3093f85f1cb6fe5c2d33523dc3a4fdbcf57755abb6b713ebb90fcb6f48`), `manifests/episodes/molmoact2_libero_spatial_transfer.json` (`ce5738c89fda65d7fc655ad67b33d9aef3e9d74ae898e188592fdcccac70687f`)
- Preset evolution launch: `launch/routes/libero/molmoact2_base_four_suite.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/molmoact2_base_four_suite.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 50 | 520 | `{"task_id":"KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}` |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 50 | 520 | `{"task_id":"KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}` |
| `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 50 | 520 | `{"task_id":"KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"}` |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 50 | 520 | `{"task_id":"KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}` |
| `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"}` |
| `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"}` |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"}` |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 50 | 520 | `{"task_id":"STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"}` |
| `open_the_middle_drawer_of_the_cabinet` | 50 | 300 | `{"task_id":"open_the_middle_drawer_of_the_cabinet"}` |
| `open_the_top_drawer_and_put_the_bowl_inside` | 50 | 300 | `{"task_id":"open_the_top_drawer_and_put_the_bowl_inside"}` |
| `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_alphabet_soup_and_place_it_in_the_basket"}` |
| `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_bbq_sauce_and_place_it_in_the_basket"}` |
| `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_butter_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_butter_and_place_it_in_the_basket"}` |
| `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_chocolate_pudding_and_place_it_in_the_basket"}` |
| `pick_up_the_cream_cheese_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_cream_cheese_and_place_it_in_the_basket"}` |
| `pick_up_the_ketchup_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_ketchup_and_place_it_in_the_basket"}` |
| `pick_up_the_milk_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_milk_and_place_it_in_the_basket"}` |
| `pick_up_the_orange_juice_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_orange_juice_and_place_it_in_the_basket"}` |
| `pick_up_the_salad_dressing_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_salad_dressing_and_place_it_in_the_basket"}` |
| `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_tomato_sauce_and_place_it_in_the_basket"}` |
| `push_the_plate_to_the_front_of_the_stove` | 50 | 300 | `{"task_id":"push_the_plate_to_the_front_of_the_stove"}` |
| `put_the_bowl_on_the_plate` | 50 | 300 | `{"task_id":"put_the_bowl_on_the_plate"}` |
| `put_the_bowl_on_the_stove` | 50 | 300 | `{"task_id":"put_the_bowl_on_the_stove"}` |
| `put_the_bowl_on_top_of_the_cabinet` | 50 | 300 | `{"task_id":"put_the_bowl_on_top_of_the_cabinet"}` |
| `put_the_cream_cheese_in_the_bowl` | 50 | 300 | `{"task_id":"put_the_cream_cheese_in_the_bowl"}` |
| `put_the_wine_bottle_on_the_rack` | 50 | 300 | `{"task_id":"put_the_wine_bottle_on_the_rack"}` |
| `put_the_wine_bottle_on_top_of_the_cabinet` | 50 | 300 | `{"task_id":"put_the_wine_bottle_on_top_of_the_cabinet"}` |
| `turn_on_the_stove` | 50 | 300 | `{"task_id":"turn_on_the_stove"}` |

## `molmoact2_think_libero`

- Route: MolmoAct2 Think + complete four-suite LIBERO
- Study role: canonical full benchmark
- Launcher: `launch/routes/libero/molmoact2_think_four_suite.sh`
- Profile: `configs/molmoact2_think_libero.json` (`89f585eaf2480edd6bcd86fff7e6efc507f9459e07fcdf1ed7bbbbacdb2ed907`)
- Profile set: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [allenai/MolmoAct2-Think-LIBERO](https://huggingface.co/allenai/MolmoAct2-Think-LIBERO/tree/593d25fcd3150e38eb05812fc3f9adb02927ec83) at `593d25fcd3150e38eb05812fc3f9adb02927ec83`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `molmoact2_libero_10_canonical_50_per_task_v1`, `molmoact2_libero_goal_canonical_50_per_task_v1`, `molmoact2_libero_object_canonical_50_per_task_v1`, `molmoact2_libero_spatial_canonical_50_per_task_v1`
- Standard route rows: 2000
- Comparability: This executes the complete standard four-suite plan with one shared evolving scaffold. The agent adds frozen tools, so it is comparable as an agent result, not as the raw policy baseline.
- Route benchmark plan: `manifests/benchmarks/molmoact2_think_libero_standard.json` (`db97900dc876df64fb5728bab51daf7b5d615a067a4a749f70bb0f20e7c5a3a9`)
- Exact standard source: `manifests/benchmarks/molmoact2_think_libero_standard.json` (`db97900dc876df64fb5728bab51daf7b5d615a067a4a749f70bb0f20e7c5a3a9`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`, `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`, `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`, `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`, `open_the_middle_drawer_of_the_cabinet`, `pick_up_the_alphabet_soup_and_place_it_in_the_basket`, `pick_up_the_bbq_sauce_and_place_it_in_the_basket`, `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_cream_cheese_and_place_it_in_the_basket`, `pick_up_the_ketchup_and_place_it_in_the_basket`, `pick_up_the_salad_dressing_and_place_it_in_the_basket`, `put_the_bowl_on_the_stove`, `put_the_bowl_on_top_of_the_cabinet`, `put_the_wine_bottle_on_top_of_the_cabinet`
- Preset held-out tasks: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`, `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`, `open_the_top_drawer_and_put_the_bowl_inside`, `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_butter_and_place_it_in_the_basket`, `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, `pick_up_the_milk_and_place_it_in_the_basket`, `pick_up_the_orange_juice_and_place_it_in_the_basket`, `pick_up_the_tomato_sauce_and_place_it_in_the_basket`, `put_the_bowl_on_the_plate`, `put_the_wine_bottle_on_the_rack`
- Preset sources: `manifests/episodes/molmoact2_libero_long_transfer.json` (`f417482fcab976342f907ae436440dc520fd613a632adc5f1940f2bc3ed8841b`), `manifests/episodes/molmoact2_libero_goal_transfer.json` (`aacbb4bf7c5008bc05187534b570da02b0283bc8b3acf9ad2b6b636f564f705d`), `manifests/episodes/molmoact2_libero_object_transfer.json` (`89310c3093f85f1cb6fe5c2d33523dc3a4fdbcf57755abb6b713ebb90fcb6f48`), `manifests/episodes/molmoact2_libero_spatial_transfer.json` (`ce5738c89fda65d7fc655ad67b33d9aef3e9d74ae898e188592fdcccac70687f`)
- Preset evolution launch: `launch/routes/libero/molmoact2_think_four_suite.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/molmoact2_think_four_suite.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 50 | 520 | `{"task_id":"KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}` |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 50 | 520 | `{"task_id":"KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}` |
| `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 50 | 520 | `{"task_id":"KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"}` |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 50 | 520 | `{"task_id":"KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}` |
| `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"}` |
| `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"}` |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"}` |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 50 | 520 | `{"task_id":"STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"}` |
| `open_the_middle_drawer_of_the_cabinet` | 50 | 300 | `{"task_id":"open_the_middle_drawer_of_the_cabinet"}` |
| `open_the_top_drawer_and_put_the_bowl_inside` | 50 | 300 | `{"task_id":"open_the_top_drawer_and_put_the_bowl_inside"}` |
| `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_alphabet_soup_and_place_it_in_the_basket"}` |
| `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_bbq_sauce_and_place_it_in_the_basket"}` |
| `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_butter_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_butter_and_place_it_in_the_basket"}` |
| `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_chocolate_pudding_and_place_it_in_the_basket"}` |
| `pick_up_the_cream_cheese_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_cream_cheese_and_place_it_in_the_basket"}` |
| `pick_up_the_ketchup_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_ketchup_and_place_it_in_the_basket"}` |
| `pick_up_the_milk_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_milk_and_place_it_in_the_basket"}` |
| `pick_up_the_orange_juice_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_orange_juice_and_place_it_in_the_basket"}` |
| `pick_up_the_salad_dressing_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_salad_dressing_and_place_it_in_the_basket"}` |
| `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_tomato_sauce_and_place_it_in_the_basket"}` |
| `push_the_plate_to_the_front_of_the_stove` | 50 | 300 | `{"task_id":"push_the_plate_to_the_front_of_the_stove"}` |
| `put_the_bowl_on_the_plate` | 50 | 300 | `{"task_id":"put_the_bowl_on_the_plate"}` |
| `put_the_bowl_on_the_stove` | 50 | 300 | `{"task_id":"put_the_bowl_on_the_stove"}` |
| `put_the_bowl_on_top_of_the_cabinet` | 50 | 300 | `{"task_id":"put_the_bowl_on_top_of_the_cabinet"}` |
| `put_the_cream_cheese_in_the_bowl` | 50 | 300 | `{"task_id":"put_the_cream_cheese_in_the_bowl"}` |
| `put_the_wine_bottle_on_the_rack` | 50 | 300 | `{"task_id":"put_the_wine_bottle_on_the_rack"}` |
| `put_the_wine_bottle_on_top_of_the_cabinet` | 50 | 300 | `{"task_id":"put_the_wine_bottle_on_top_of_the_cabinet"}` |
| `turn_on_the_stove` | 50 | 300 | `{"task_id":"turn_on_the_stove"}` |

## `pi05_libero`

- Route: LeRobot pi0.5 + complete four-suite LIBERO
- Study role: canonical full benchmark
- Launcher: `launch/routes/libero/lerobot_pi05_four_suite.sh`
- Profile: `configs/pi05_libero.json` (`6d0357e2f84bbb4d9248551fb8549dd091f4d78e24f859f8756362e0b08f0a54`)
- Profile set: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [lerobot/pi05_libero_finetuned](https://huggingface.co/lerobot/pi05_libero_finetuned/tree/dbf8a3f794a9c4297b44f40b752712f50073d945) at `dbf8a3f794a9c4297b44f40b752712f50073d945`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `pi05_lerobot_libero_10_canonical_10_per_task_v1`, `pi05_lerobot_libero_goal_canonical_10_per_task_v1`, `pi05_lerobot_libero_object_canonical_10_per_task_v1`, `pi05_lerobot_libero_spatial_canonical_10_per_task_v1`
- Standard route rows: 400
- Comparability: This executes the complete standard four-suite plan with one shared evolving scaffold. The agent adds frozen tools, so it is comparable as an agent result, not as the raw policy baseline.
- Route benchmark plan: `manifests/benchmarks/pi05_libero_standard.json` (`d05e9572a9f5f13b545c9a8cea2e3de4bd9adf09347065c76a65184837e62d72`)
- Exact standard source: `manifests/benchmarks/pi05_libero_standard.json` (`d05e9572a9f5f13b545c9a8cea2e3de4bd9adf09347065c76a65184837e62d72`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`, `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`, `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`, `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`, `open_the_middle_drawer_of_the_cabinet`, `pick_up_the_alphabet_soup_and_place_it_in_the_basket`, `pick_up_the_bbq_sauce_and_place_it_in_the_basket`, `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_cream_cheese_and_place_it_in_the_basket`, `pick_up_the_ketchup_and_place_it_in_the_basket`, `pick_up_the_salad_dressing_and_place_it_in_the_basket`, `put_the_bowl_on_the_stove`, `put_the_bowl_on_top_of_the_cabinet`, `put_the_wine_bottle_on_top_of_the_cabinet`
- Preset held-out tasks: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`, `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`, `open_the_top_drawer_and_put_the_bowl_inside`, `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_butter_and_place_it_in_the_basket`, `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, `pick_up_the_milk_and_place_it_in_the_basket`, `pick_up_the_orange_juice_and_place_it_in_the_basket`, `pick_up_the_tomato_sauce_and_place_it_in_the_basket`, `put_the_bowl_on_the_plate`, `put_the_wine_bottle_on_the_rack`
- Preset sources: `manifests/episodes/pi05_libero_long_transfer.json` (`3500b3f11ec445bb36a40f34fcd7c048571a9bc1cad9a2c5b84f9393a594b347`), `manifests/episodes/pi05_libero_goal_transfer.json` (`30372ad39420899d9032fb864da058529340335d88b9663ee97cbb7210bbe846`), `manifests/episodes/pi05_libero_object_transfer.json` (`d6638ca261f40008574b1164cd98ea1edd53e75e33b8d501fd37bf1fcd40559f`), `manifests/episodes/pi05_libero_spatial_transfer.json` (`5cda1aa7c207cbda2303ed228e59c2d164b9ba5957de75e721f3ddf31fb76c6a`)
- Preset evolution launch: `launch/routes/libero/lerobot_pi05_four_suite.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/lerobot_pi05_four_suite.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 10 | 520 | `{"task_id":"KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}` |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 10 | 520 | `{"task_id":"KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}` |
| `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 10 | 520 | `{"task_id":"KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"}` |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 10 | 520 | `{"task_id":"KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}` |
| `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"}` |
| `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"}` |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"}` |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 10 | 520 | `{"task_id":"STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"}` |
| `open_the_middle_drawer_of_the_cabinet` | 10 | 300 | `{"task_id":"open_the_middle_drawer_of_the_cabinet"}` |
| `open_the_top_drawer_and_put_the_bowl_inside` | 10 | 300 | `{"task_id":"open_the_top_drawer_and_put_the_bowl_inside"}` |
| `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_alphabet_soup_and_place_it_in_the_basket"}` |
| `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_bbq_sauce_and_place_it_in_the_basket"}` |
| `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_butter_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_butter_and_place_it_in_the_basket"}` |
| `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_chocolate_pudding_and_place_it_in_the_basket"}` |
| `pick_up_the_cream_cheese_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_cream_cheese_and_place_it_in_the_basket"}` |
| `pick_up_the_ketchup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_ketchup_and_place_it_in_the_basket"}` |
| `pick_up_the_milk_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_milk_and_place_it_in_the_basket"}` |
| `pick_up_the_orange_juice_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_orange_juice_and_place_it_in_the_basket"}` |
| `pick_up_the_salad_dressing_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_salad_dressing_and_place_it_in_the_basket"}` |
| `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_tomato_sauce_and_place_it_in_the_basket"}` |
| `push_the_plate_to_the_front_of_the_stove` | 10 | 300 | `{"task_id":"push_the_plate_to_the_front_of_the_stove"}` |
| `put_the_bowl_on_the_plate` | 10 | 300 | `{"task_id":"put_the_bowl_on_the_plate"}` |
| `put_the_bowl_on_the_stove` | 10 | 300 | `{"task_id":"put_the_bowl_on_the_stove"}` |
| `put_the_bowl_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"put_the_bowl_on_top_of_the_cabinet"}` |
| `put_the_cream_cheese_in_the_bowl` | 10 | 300 | `{"task_id":"put_the_cream_cheese_in_the_bowl"}` |
| `put_the_wine_bottle_on_the_rack` | 10 | 300 | `{"task_id":"put_the_wine_bottle_on_the_rack"}` |
| `put_the_wine_bottle_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"put_the_wine_bottle_on_top_of_the_cabinet"}` |
| `turn_on_the_stove` | 10 | 300 | `{"task_id":"turn_on_the_stove"}` |

## `rlinf_pi05_libero`

- Route: RLinf pi0.5 + complete four-suite LIBERO
- Study role: canonical full benchmark
- Launcher: `launch/routes/libero/rlinf_pi05_four_suite.sh`
- Profile: `configs/rlinf_pi05_libero_spatial.json` (`9b1fc038007ab72940408dfc3fd51ba43a5b48d46e7f7e6848cf2ccde4c1d602`)
- Profile set: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT/tree/6222623f635769bfc73c9472e29fab9b7fd8e027) at `6222623f635769bfc73c9472e29fab9b7fd8e027`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rlinf_pi05_libero_goal_canonical_10_per_task_v1`, `rlinf_pi05_libero_long_canonical_10_per_task_v1`, `rlinf_pi05_libero_object_canonical_10_per_task_v1`, `rlinf_pi05_libero_spatial_canonical_10_per_task_v1`
- Standard route rows: 400
- Comparability: This executes the complete standard four-suite plan with one shared evolving scaffold. The agent adds frozen tools, so it is comparable as an agent result, not as the raw policy baseline.
- Route benchmark plan: `manifests/benchmarks/rlinf_pi05_libero_standard.json` (`ebf9966972d174408d6563e380b82d6d7c3b2438723d8f459a733c1c3cad3e55`)
- Exact standard source: `manifests/benchmarks/rlinf_pi05_libero_standard.json` (`ebf9966972d174408d6563e380b82d6d7c3b2438723d8f459a733c1c3cad3e55`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`, `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`, `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`, `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`, `open_the_middle_drawer_of_the_cabinet`, `pick_up_the_alphabet_soup_and_place_it_in_the_basket`, `pick_up_the_bbq_sauce_and_place_it_in_the_basket`, `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_cream_cheese_and_place_it_in_the_basket`, `pick_up_the_ketchup_and_place_it_in_the_basket`, `pick_up_the_salad_dressing_and_place_it_in_the_basket`, `put_the_bowl_on_the_stove`, `put_the_bowl_on_top_of_the_cabinet`, `put_the_wine_bottle_on_top_of_the_cabinet`
- Preset held-out tasks: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`, `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`, `open_the_top_drawer_and_put_the_bowl_inside`, `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_butter_and_place_it_in_the_basket`, `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, `pick_up_the_milk_and_place_it_in_the_basket`, `pick_up_the_orange_juice_and_place_it_in_the_basket`, `pick_up_the_tomato_sauce_and_place_it_in_the_basket`, `put_the_bowl_on_the_plate`, `put_the_wine_bottle_on_the_rack`
- Preset sources: `manifests/episodes/rlinf_pi05_libero_long_related_transfer.json` (`031bbe1037d8ca788028bb37a57302a8b58a18a332f38b76b86d51a35f7acda9`), `manifests/episodes/rlinf_pi05_libero_goal_related_transfer.json` (`a8fc8d2e9d1bde32022492ebe7d13afbe24be4e64b2c31a09f7ac3bfe07de3a9`), `manifests/episodes/rlinf_pi05_libero_object_related_transfer.json` (`f0317a82779d3f6793d4039b196e964605d07c4d037969cf75141a853568280b`), `manifests/episodes/rlinf_pi05_libero_spatial_related_transfer.json` (`db338b2b36072f80ec9d39cbef847a3d5d0aae6a4f3c3a4c7d92b97f2aeee50a`)
- Preset evolution launch: `launch/routes/libero/rlinf_pi05_four_suite.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/rlinf_pi05_four_suite.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 10 | 520 | `{"task_id":"KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}` |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 10 | 520 | `{"task_id":"KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}` |
| `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 10 | 520 | `{"task_id":"KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"}` |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 10 | 520 | `{"task_id":"KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}` |
| `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"}` |
| `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"}` |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"}` |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 10 | 520 | `{"task_id":"STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"}` |
| `open_the_middle_drawer_of_the_cabinet` | 10 | 300 | `{"task_id":"open_the_middle_drawer_of_the_cabinet"}` |
| `open_the_top_drawer_and_put_the_bowl_inside` | 10 | 300 | `{"task_id":"open_the_top_drawer_and_put_the_bowl_inside"}` |
| `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_alphabet_soup_and_place_it_in_the_basket"}` |
| `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_bbq_sauce_and_place_it_in_the_basket"}` |
| `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_butter_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_butter_and_place_it_in_the_basket"}` |
| `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_chocolate_pudding_and_place_it_in_the_basket"}` |
| `pick_up_the_cream_cheese_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_cream_cheese_and_place_it_in_the_basket"}` |
| `pick_up_the_ketchup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_ketchup_and_place_it_in_the_basket"}` |
| `pick_up_the_milk_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_milk_and_place_it_in_the_basket"}` |
| `pick_up_the_orange_juice_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_orange_juice_and_place_it_in_the_basket"}` |
| `pick_up_the_salad_dressing_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_salad_dressing_and_place_it_in_the_basket"}` |
| `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_tomato_sauce_and_place_it_in_the_basket"}` |
| `push_the_plate_to_the_front_of_the_stove` | 10 | 300 | `{"task_id":"push_the_plate_to_the_front_of_the_stove"}` |
| `put_the_bowl_on_the_plate` | 10 | 300 | `{"task_id":"put_the_bowl_on_the_plate"}` |
| `put_the_bowl_on_the_stove` | 10 | 300 | `{"task_id":"put_the_bowl_on_the_stove"}` |
| `put_the_bowl_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"put_the_bowl_on_top_of_the_cabinet"}` |
| `put_the_cream_cheese_in_the_bowl` | 10 | 300 | `{"task_id":"put_the_cream_cheese_in_the_bowl"}` |
| `put_the_wine_bottle_on_the_rack` | 10 | 300 | `{"task_id":"put_the_wine_bottle_on_the_rack"}` |
| `put_the_wine_bottle_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"put_the_wine_bottle_on_top_of_the_cabinet"}` |
| `turn_on_the_stove` | 10 | 300 | `{"task_id":"turn_on_the_stove"}` |

## `rlinf_pi05_libero_pro`

- Route: RLinf pi0.5 + complete eight-cell LIBERO-Pro
- Study role: canonical full benchmark
- Launcher: `launch/routes/libero_pro/rlinf_pi05_eight_cell.sh`
- Profile: `configs/rlinf_pi05_libero_pro_spatial_task.json` (`1e19bcf9da9f3e82f4aebd6de7c86945b3cde514f8b881a2cd5ef1be5085e8d5`)
- Profile set: `libero_pro_spatial_task`, `libero_pro_spatial_swap`, `libero_pro_object_task`, `libero_pro_object_swap`, `libero_pro_goal_task`, `libero_pro_goal_swap`, `libero_pro_10_task`, `libero_pro_10_swap`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT/tree/6222623f635769bfc73c9472e29fab9b7fd8e027) at `6222623f635769bfc73c9472e29fab9b7fd8e027`
- Full benchmark status: `ready`
- Metric: `equal_cell_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rlinf_pi05_libero_pro_10_swap_paper_v3_10_seed_v1`, `rlinf_pi05_libero_pro_10_task_paper_v3_10_seed_v1`, `rlinf_pi05_libero_pro_goal_swap_paper_v3_10_seed_v1`, `rlinf_pi05_libero_pro_goal_task_paper_v3_10_seed_v1`, `rlinf_pi05_libero_pro_object_swap_paper_v3_10_seed_v1`, `rlinf_pi05_libero_pro_object_task_paper_v3_10_seed_v1`, `rlinf_pi05_libero_pro_spatial_swap_paper_v3_10_seed_v1`, `rlinf_pi05_libero_pro_spatial_task_paper_v3_10_seed_v1`
- Standard route rows: 800
- Comparability: This executes the complete Harness paper-v3 eight-cell plan with one shared evolving scaffold. It uses the released RLinf policy without the unreleased Harness memory agent.
- Route benchmark plan: `manifests/benchmarks/rlinf_pi05_libero_pro_harness_paper_v3.json` (`03e6adde51c602740f3bf5c9f3d0e55640458ffe5d88d472c70ba730f78f0412`)
- Exact standard source: `manifests/benchmarks/rlinf_pi05_libero_pro_harness_paper_v3.json` (`03e6adde51c602740f3bf5c9f3d0e55640458ffe5d88d472c70ba730f78f0412`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `libero_pro_10_swap::KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `libero_pro_10_swap::KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`, `libero_pro_10_swap::LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`, `libero_pro_10_swap::LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`, `libero_pro_10_swap::STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`, `libero_pro_10_task::KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `libero_pro_10_task::KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`, `libero_pro_10_task::LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`, `libero_pro_10_task::LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`, `libero_pro_10_task::STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`, `libero_pro_goal_swap::open_the_middle_drawer_of_the_cabinet`, `libero_pro_goal_swap::open_the_top_drawer_and_put_the_bowl_inside`, `libero_pro_goal_swap::put_the_bowl_on_the_stove`, `libero_pro_goal_swap::put_the_wine_bottle_on_top_of_the_cabinet`, `libero_pro_goal_swap::turn_on_the_stove`, `libero_pro_goal_task::open_the_middle_drawer_of_the_cabinet`, `libero_pro_goal_task::push_the_plate_to_the_front_of_the_stove`, `libero_pro_goal_task::put_the_cream_cheese_in_the_bowl`, `libero_pro_goal_task::put_the_wine_bottle_on_top_of_the_cabinet`, `libero_pro_goal_task::turn_on_the_stove`, `libero_pro_object_swap::pick_up_the_alphabet_soup_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_bbq_sauce_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_cream_cheese_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_ketchup_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_salad_dressing_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_alphabet_soup_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_bbq_sauce_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_cream_cheese_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_ketchup_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_salad_dressing_and_place_it_in_the_basket`, `libero_pro_spatial_swap::pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`
- Preset held-out tasks: `libero_pro_10_swap::KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`, `libero_pro_10_swap::KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, `libero_pro_10_swap::LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, `libero_pro_10_swap::LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`, `libero_pro_10_swap::LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`, `libero_pro_10_task::KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`, `libero_pro_10_task::KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, `libero_pro_10_task::LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, `libero_pro_10_task::LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`, `libero_pro_10_task::LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`, `libero_pro_goal_swap::push_the_plate_to_the_front_of_the_stove`, `libero_pro_goal_swap::put_the_bowl_on_the_plate`, `libero_pro_goal_swap::put_the_bowl_on_top_of_the_cabinet`, `libero_pro_goal_swap::put_the_cream_cheese_in_the_bowl`, `libero_pro_goal_swap::put_the_wine_bottle_on_the_rack`, `libero_pro_goal_task::open_the_top_drawer_and_put_the_bowl_inside`, `libero_pro_goal_task::put_the_bowl_on_the_plate`, `libero_pro_goal_task::put_the_bowl_on_the_stove`, `libero_pro_goal_task::put_the_bowl_on_top_of_the_cabinet`, `libero_pro_goal_task::put_the_wine_bottle_on_the_rack`, `libero_pro_object_swap::pick_up_the_butter_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_milk_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_orange_juice_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_tomato_sauce_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_butter_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_milk_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_orange_juice_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_tomato_sauce_and_place_it_in_the_basket`, `libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`
- Preset sources: `manifests/episodes/rlinf_pi05_libero_pro_10_swap_related_transfer.json` (`ea7985e243569c264752e3d823226df6d7104880c6b3c94f2c9d918101d1c711`), `manifests/episodes/rlinf_pi05_libero_pro_10_task_related_transfer.json` (`54ad0bdb3acb5420a051640a0c98383936a27a80ffc8db8ffc9a5a3714707319`), `manifests/episodes/rlinf_pi05_libero_pro_goal_swap_related_transfer.json` (`afc3125c3cb074b63d57d2f14ab5758011a2f3ca9c7c2ed984f8104e002aa9db`), `manifests/episodes/rlinf_pi05_libero_pro_goal_task_related_transfer.json` (`97ac796f83b8aa5a8a59e5190a36d879668cdaeee906f5bc037624502491ac7d`), `manifests/episodes/rlinf_pi05_libero_pro_object_swap_related_transfer.json` (`d9f962e5eb781df2080617812bbf51d1792dc7e00952073a732ec24bd6addc1a`), `manifests/episodes/rlinf_pi05_libero_pro_object_task_related_transfer.json` (`ca972e3d7e8bb9ec357fbfd35d36744e03fd62b1cd3d4e274913307ca5f52dea`), `manifests/episodes/rlinf_pi05_libero_pro_spatial_swap_related_transfer.json` (`407ec0df81a1886d00995b287d60fbc5b64ea259a9f1a498b7ccc339ffcf660a`), `manifests/episodes/rlinf_pi05_libero_pro_spatial_task_related_transfer.json` (`a8cd409dd8868fb5208df82acff5e4991ab7d2f519a21157fb7b7c8b4038fd41`)
- Preset evolution launch: `launch/routes/libero_pro/rlinf_pi05_eight_cell.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero_pro/rlinf_pi05_eight_cell.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `libero_pro_10_swap::KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 10 | 520 | `{"task_id":"libero_pro_10_swap::KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}` |
| `libero_pro_10_swap::KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 10 | 520 | `{"task_id":"libero_pro_10_swap::KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}` |
| `libero_pro_10_swap::KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 10 | 520 | `{"task_id":"libero_pro_10_swap::KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"}` |
| `libero_pro_10_swap::KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 10 | 520 | `{"task_id":"libero_pro_10_swap::KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}` |
| `libero_pro_10_swap::LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 10 | 520 | `{"task_id":"libero_pro_10_swap::LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"}` |
| `libero_pro_10_swap::LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 10 | 520 | `{"task_id":"libero_pro_10_swap::LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"}` |
| `libero_pro_10_swap::LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 10 | 520 | `{"task_id":"libero_pro_10_swap::LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"}` |
| `libero_pro_10_swap::LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 10 | 520 | `{"task_id":"libero_pro_10_swap::LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"}` |
| `libero_pro_10_swap::LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 10 | 520 | `{"task_id":"libero_pro_10_swap::LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"}` |
| `libero_pro_10_swap::STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 10 | 520 | `{"task_id":"libero_pro_10_swap::STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"}` |
| `libero_pro_10_task::KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 10 | 520 | `{"task_id":"libero_pro_10_task::KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}` |
| `libero_pro_10_task::KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 10 | 520 | `{"task_id":"libero_pro_10_task::KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}` |
| `libero_pro_10_task::KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 10 | 520 | `{"task_id":"libero_pro_10_task::KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"}` |
| `libero_pro_10_task::KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 10 | 520 | `{"task_id":"libero_pro_10_task::KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}` |
| `libero_pro_10_task::LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 10 | 520 | `{"task_id":"libero_pro_10_task::LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"}` |
| `libero_pro_10_task::LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 10 | 520 | `{"task_id":"libero_pro_10_task::LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"}` |
| `libero_pro_10_task::LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 10 | 520 | `{"task_id":"libero_pro_10_task::LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"}` |
| `libero_pro_10_task::LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 10 | 520 | `{"task_id":"libero_pro_10_task::LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"}` |
| `libero_pro_10_task::LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 10 | 520 | `{"task_id":"libero_pro_10_task::LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"}` |
| `libero_pro_10_task::STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 10 | 520 | `{"task_id":"libero_pro_10_task::STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"}` |
| `libero_pro_goal_swap::open_the_middle_drawer_of_the_cabinet` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::open_the_middle_drawer_of_the_cabinet"}` |
| `libero_pro_goal_swap::open_the_top_drawer_and_put_the_bowl_inside` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::open_the_top_drawer_and_put_the_bowl_inside"}` |
| `libero_pro_goal_swap::push_the_plate_to_the_front_of_the_stove` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::push_the_plate_to_the_front_of_the_stove"}` |
| `libero_pro_goal_swap::put_the_bowl_on_the_plate` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::put_the_bowl_on_the_plate"}` |
| `libero_pro_goal_swap::put_the_bowl_on_the_stove` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::put_the_bowl_on_the_stove"}` |
| `libero_pro_goal_swap::put_the_bowl_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::put_the_bowl_on_top_of_the_cabinet"}` |
| `libero_pro_goal_swap::put_the_cream_cheese_in_the_bowl` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::put_the_cream_cheese_in_the_bowl"}` |
| `libero_pro_goal_swap::put_the_wine_bottle_on_the_rack` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::put_the_wine_bottle_on_the_rack"}` |
| `libero_pro_goal_swap::put_the_wine_bottle_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::put_the_wine_bottle_on_top_of_the_cabinet"}` |
| `libero_pro_goal_swap::turn_on_the_stove` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::turn_on_the_stove"}` |
| `libero_pro_goal_task::open_the_middle_drawer_of_the_cabinet` | 10 | 300 | `{"task_id":"libero_pro_goal_task::open_the_middle_drawer_of_the_cabinet"}` |
| `libero_pro_goal_task::open_the_top_drawer_and_put_the_bowl_inside` | 10 | 300 | `{"task_id":"libero_pro_goal_task::open_the_top_drawer_and_put_the_bowl_inside"}` |
| `libero_pro_goal_task::push_the_plate_to_the_front_of_the_stove` | 10 | 300 | `{"task_id":"libero_pro_goal_task::push_the_plate_to_the_front_of_the_stove"}` |
| `libero_pro_goal_task::put_the_bowl_on_the_plate` | 10 | 300 | `{"task_id":"libero_pro_goal_task::put_the_bowl_on_the_plate"}` |
| `libero_pro_goal_task::put_the_bowl_on_the_stove` | 10 | 300 | `{"task_id":"libero_pro_goal_task::put_the_bowl_on_the_stove"}` |
| `libero_pro_goal_task::put_the_bowl_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"libero_pro_goal_task::put_the_bowl_on_top_of_the_cabinet"}` |
| `libero_pro_goal_task::put_the_cream_cheese_in_the_bowl` | 10 | 300 | `{"task_id":"libero_pro_goal_task::put_the_cream_cheese_in_the_bowl"}` |
| `libero_pro_goal_task::put_the_wine_bottle_on_the_rack` | 10 | 300 | `{"task_id":"libero_pro_goal_task::put_the_wine_bottle_on_the_rack"}` |
| `libero_pro_goal_task::put_the_wine_bottle_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"libero_pro_goal_task::put_the_wine_bottle_on_top_of_the_cabinet"}` |
| `libero_pro_goal_task::turn_on_the_stove` | 10 | 300 | `{"task_id":"libero_pro_goal_task::turn_on_the_stove"}` |
| `libero_pro_object_swap::pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_alphabet_soup_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_bbq_sauce_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_butter_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_butter_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_chocolate_pudding_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_cream_cheese_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_cream_cheese_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_ketchup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_ketchup_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_milk_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_milk_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_orange_juice_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_orange_juice_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_salad_dressing_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_salad_dressing_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_tomato_sauce_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_alphabet_soup_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_bbq_sauce_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_butter_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_butter_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_chocolate_pudding_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_cream_cheese_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_cream_cheese_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_ketchup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_ketchup_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_milk_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_milk_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_orange_juice_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_orange_juice_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_salad_dressing_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_salad_dressing_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_tomato_sauce_and_place_it_in_the_basket"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}` |

## `xvla_libero`

- Route: X-VLA + complete four-suite LIBERO
- Study role: canonical full benchmark
- Launcher: `launch/routes/libero/xvla_four_suite.sh`
- Profile: `configs/xvla_libero.json` (`f28799a2524dc56316b5373cf9d43a573a56d13ea69b2d80d636b5b2452ad7c4`)
- Profile set: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [2toINF/X-VLA-Libero](https://huggingface.co/2toINF/X-VLA-Libero/tree/129e71460678b7236cee6fc9707f09d9fa0c3590) at `129e71460678b7236cee6fc9707f09d9fa0c3590`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `xvla_libero_10_canonical_50_per_task_v1`, `xvla_libero_goal_canonical_50_per_task_v1`, `xvla_libero_object_canonical_50_per_task_v1`, `xvla_libero_spatial_canonical_50_per_task_v1`
- Standard route rows: 2000
- Comparability: This executes the complete standard four-suite plan with one shared evolving scaffold. The agent adds frozen tools, so it is comparable as an agent result, not as the raw policy baseline.
- Route benchmark plan: `manifests/benchmarks/xvla_libero_standard.json` (`c9f2aa2715e983c81e82cc9458ce494477caf025b55d6cefc25e4e3ba250a930`)
- Exact standard source: `manifests/benchmarks/xvla_libero_standard.json` (`c9f2aa2715e983c81e82cc9458ce494477caf025b55d6cefc25e4e3ba250a930`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`, `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`, `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`, `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`, `open_the_middle_drawer_of_the_cabinet`, `pick_up_the_alphabet_soup_and_place_it_in_the_basket`, `pick_up_the_bbq_sauce_and_place_it_in_the_basket`, `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_cream_cheese_and_place_it_in_the_basket`, `pick_up_the_ketchup_and_place_it_in_the_basket`, `pick_up_the_salad_dressing_and_place_it_in_the_basket`, `put_the_bowl_on_the_stove`, `put_the_bowl_on_top_of_the_cabinet`, `put_the_wine_bottle_on_top_of_the_cabinet`
- Preset held-out tasks: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`, `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`, `open_the_top_drawer_and_put_the_bowl_inside`, `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_butter_and_place_it_in_the_basket`, `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, `pick_up_the_milk_and_place_it_in_the_basket`, `pick_up_the_orange_juice_and_place_it_in_the_basket`, `pick_up_the_tomato_sauce_and_place_it_in_the_basket`, `put_the_bowl_on_the_plate`, `put_the_wine_bottle_on_the_rack`
- Preset sources: `manifests/episodes/xvla_libero_long_transfer.json` (`61a16159a8281facb36c9b78cf7f098fdc4ae98c6871ff2747c96a6efb8c294c`), `manifests/episodes/xvla_libero_goal_transfer.json` (`146fe60e2fceeb4545325f863233585edaadddc9f572e33a85edf632b378fb2c`), `manifests/episodes/xvla_libero_object_transfer.json` (`9aa0aa049197200cad8d0c3bae1fca538a415eca7d97569c47d4adabf3d056e0`), `manifests/episodes/libero_spatial_transfer.json` (`5fb3ab13c040b78afefbd251e882e90d6b3bc6e013759a0c88ed7ecfc00da26c`)
- Preset evolution launch: `launch/routes/libero/xvla_four_suite.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/xvla_four_suite.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 50 | 900 | `{"task_id":"KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}` |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 50 | 900 | `{"task_id":"KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}` |
| `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 50 | 900 | `{"task_id":"KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"}` |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 50 | 900 | `{"task_id":"KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}` |
| `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 50 | 900 | `{"task_id":"LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 50 | 900 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 50 | 900 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"}` |
| `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 50 | 900 | `{"task_id":"LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"}` |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 50 | 900 | `{"task_id":"LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"}` |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 50 | 900 | `{"task_id":"STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"}` |
| `open_the_middle_drawer_of_the_cabinet` | 50 | 800 | `{"task_id":"open_the_middle_drawer_of_the_cabinet"}` |
| `open_the_top_drawer_and_put_the_bowl_inside` | 50 | 800 | `{"task_id":"open_the_top_drawer_and_put_the_bowl_inside"}` |
| `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_alphabet_soup_and_place_it_in_the_basket"}` |
| `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_bbq_sauce_and_place_it_in_the_basket"}` |
| `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_butter_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_butter_and_place_it_in_the_basket"}` |
| `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_chocolate_pudding_and_place_it_in_the_basket"}` |
| `pick_up_the_cream_cheese_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_cream_cheese_and_place_it_in_the_basket"}` |
| `pick_up_the_ketchup_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_ketchup_and_place_it_in_the_basket"}` |
| `pick_up_the_milk_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_milk_and_place_it_in_the_basket"}` |
| `pick_up_the_orange_juice_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_orange_juice_and_place_it_in_the_basket"}` |
| `pick_up_the_salad_dressing_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_salad_dressing_and_place_it_in_the_basket"}` |
| `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_tomato_sauce_and_place_it_in_the_basket"}` |
| `push_the_plate_to_the_front_of_the_stove` | 50 | 800 | `{"task_id":"push_the_plate_to_the_front_of_the_stove"}` |
| `put_the_bowl_on_the_plate` | 50 | 800 | `{"task_id":"put_the_bowl_on_the_plate"}` |
| `put_the_bowl_on_the_stove` | 50 | 800 | `{"task_id":"put_the_bowl_on_the_stove"}` |
| `put_the_bowl_on_top_of_the_cabinet` | 50 | 800 | `{"task_id":"put_the_bowl_on_top_of_the_cabinet"}` |
| `put_the_cream_cheese_in_the_bowl` | 50 | 800 | `{"task_id":"put_the_cream_cheese_in_the_bowl"}` |
| `put_the_wine_bottle_on_the_rack` | 50 | 800 | `{"task_id":"put_the_wine_bottle_on_the_rack"}` |
| `put_the_wine_bottle_on_top_of_the_cabinet` | 50 | 800 | `{"task_id":"put_the_wine_bottle_on_top_of_the_cabinet"}` |
| `turn_on_the_stove` | 50 | 800 | `{"task_id":"turn_on_the_stove"}` |

## `molmoact2_libero_goal`

- Route: MolmoAct2 Base + LIBERO Goal
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/molmoact2_base_goal.sh`
- Profile: `configs/molmoact2_libero_goal.json` (`00e581c5368a3b64fd8dae4f2254785bd3647eb3d67b83d7945db04fb7f59d19`)
- Profile set: `libero_goal`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [allenai/MolmoAct2-LIBERO](https://huggingface.co/allenai/MolmoAct2-LIBERO/tree/0d24a92bd1faf321ef497c3bbd5681af97c65aa2) at `0d24a92bd1faf321ef497c3bbd5681af97c65aa2`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `molmoact2_libero_goal_canonical_50_per_task_v1`
- Standard route rows: 500
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/molmoact2_libero_goal/benchmark_plan.json` (`3db52a73f4e2f49fcb2c60d463e3ec83ec83fb29ecb67cdb2492232154e22f14`)
- Exact standard source: `manifests/benchmarks/molmoact2_libero_standard.json` (`965d82f2a695e567a3bb06946b3428769c4a6aaea12790b5c9048ad0eaf51527`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `open_the_middle_drawer_of_the_cabinet`, `put_the_bowl_on_the_stove`, `put_the_bowl_on_top_of_the_cabinet`, `put_the_wine_bottle_on_top_of_the_cabinet`
- Preset held-out tasks: `open_the_top_drawer_and_put_the_bowl_inside`, `put_the_bowl_on_the_plate`, `put_the_wine_bottle_on_the_rack`
- Preset sources: `manifests/episodes/molmoact2_libero_goal_transfer.json` (`aacbb4bf7c5008bc05187534b570da02b0283bc8b3acf9ad2b6b636f564f705d`)
- Preset evolution launch: `launch/routes/libero/molmoact2_base_goal.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/molmoact2_base_goal.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `open_the_middle_drawer_of_the_cabinet` | 50 | 300 | `{"task_id":"open_the_middle_drawer_of_the_cabinet"}` |
| `open_the_top_drawer_and_put_the_bowl_inside` | 50 | 300 | `{"task_id":"open_the_top_drawer_and_put_the_bowl_inside"}` |
| `push_the_plate_to_the_front_of_the_stove` | 50 | 300 | `{"task_id":"push_the_plate_to_the_front_of_the_stove"}` |
| `put_the_bowl_on_the_plate` | 50 | 300 | `{"task_id":"put_the_bowl_on_the_plate"}` |
| `put_the_bowl_on_the_stove` | 50 | 300 | `{"task_id":"put_the_bowl_on_the_stove"}` |
| `put_the_bowl_on_top_of_the_cabinet` | 50 | 300 | `{"task_id":"put_the_bowl_on_top_of_the_cabinet"}` |
| `put_the_cream_cheese_in_the_bowl` | 50 | 300 | `{"task_id":"put_the_cream_cheese_in_the_bowl"}` |
| `put_the_wine_bottle_on_the_rack` | 50 | 300 | `{"task_id":"put_the_wine_bottle_on_the_rack"}` |
| `put_the_wine_bottle_on_top_of_the_cabinet` | 50 | 300 | `{"task_id":"put_the_wine_bottle_on_top_of_the_cabinet"}` |
| `turn_on_the_stove` | 50 | 300 | `{"task_id":"turn_on_the_stove"}` |

## `molmoact2_libero_long`

- Route: MolmoAct2 Base + LIBERO Long
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/molmoact2_base_long.sh`
- Profile: `configs/molmoact2_libero_long.json` (`345ab5e4f4baa87abe6dc40c83aff51485f90dc0a2db7228d6ff4cf13bf5d8dc`)
- Profile set: `libero_10`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [allenai/MolmoAct2-LIBERO](https://huggingface.co/allenai/MolmoAct2-LIBERO/tree/0d24a92bd1faf321ef497c3bbd5681af97c65aa2) at `0d24a92bd1faf321ef497c3bbd5681af97c65aa2`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `molmoact2_libero_10_canonical_50_per_task_v1`
- Standard route rows: 500
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/molmoact2_libero_long/benchmark_plan.json` (`bbe61be455da8179c98e4bd16835e903c34f7b4b4eabc533d4581251de153811`)
- Exact standard source: `manifests/benchmarks/molmoact2_libero_standard.json` (`965d82f2a695e567a3bb06946b3428769c4a6aaea12790b5c9048ad0eaf51527`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`, `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`, `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`, `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
- Preset held-out tasks: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`, `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`
- Preset sources: `manifests/episodes/molmoact2_libero_long_transfer.json` (`f417482fcab976342f907ae436440dc520fd613a632adc5f1940f2bc3ed8841b`)
- Preset evolution launch: `launch/routes/libero/molmoact2_base_long.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/molmoact2_base_long.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 50 | 520 | `{"task_id":"KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}` |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 50 | 520 | `{"task_id":"KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}` |
| `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 50 | 520 | `{"task_id":"KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"}` |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 50 | 520 | `{"task_id":"KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}` |
| `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"}` |
| `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"}` |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"}` |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 50 | 520 | `{"task_id":"STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"}` |

## `molmoact2_libero_object`

- Route: MolmoAct2 Base + LIBERO Object
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/molmoact2_base_object.sh`
- Profile: `configs/molmoact2_libero_object.json` (`9fb13899e7f442fa11d980bc95d1e432d53f496119645ad4d42bad41c02419f1`)
- Profile set: `libero_object`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [allenai/MolmoAct2-LIBERO](https://huggingface.co/allenai/MolmoAct2-LIBERO/tree/0d24a92bd1faf321ef497c3bbd5681af97c65aa2) at `0d24a92bd1faf321ef497c3bbd5681af97c65aa2`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `molmoact2_libero_object_canonical_50_per_task_v1`
- Standard route rows: 500
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/molmoact2_libero_object/benchmark_plan.json` (`1f7849f4d89fb70108029176203f63be96d3cd1e048d6e33c9f04542418c17ab`)
- Exact standard source: `manifests/benchmarks/molmoact2_libero_standard.json` (`965d82f2a695e567a3bb06946b3428769c4a6aaea12790b5c9048ad0eaf51527`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`, `pick_up_the_bbq_sauce_and_place_it_in_the_basket`, `pick_up_the_cream_cheese_and_place_it_in_the_basket`, `pick_up_the_ketchup_and_place_it_in_the_basket`, `pick_up_the_salad_dressing_and_place_it_in_the_basket`
- Preset held-out tasks: `pick_up_the_butter_and_place_it_in_the_basket`, `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, `pick_up_the_milk_and_place_it_in_the_basket`, `pick_up_the_orange_juice_and_place_it_in_the_basket`, `pick_up_the_tomato_sauce_and_place_it_in_the_basket`
- Preset sources: `manifests/episodes/molmoact2_libero_object_transfer.json` (`89310c3093f85f1cb6fe5c2d33523dc3a4fdbcf57755abb6b713ebb90fcb6f48`)
- Preset evolution launch: `launch/routes/libero/molmoact2_base_object.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/molmoact2_base_object.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_alphabet_soup_and_place_it_in_the_basket"}` |
| `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_bbq_sauce_and_place_it_in_the_basket"}` |
| `pick_up_the_butter_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_butter_and_place_it_in_the_basket"}` |
| `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_chocolate_pudding_and_place_it_in_the_basket"}` |
| `pick_up_the_cream_cheese_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_cream_cheese_and_place_it_in_the_basket"}` |
| `pick_up_the_ketchup_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_ketchup_and_place_it_in_the_basket"}` |
| `pick_up_the_milk_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_milk_and_place_it_in_the_basket"}` |
| `pick_up_the_orange_juice_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_orange_juice_and_place_it_in_the_basket"}` |
| `pick_up_the_salad_dressing_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_salad_dressing_and_place_it_in_the_basket"}` |
| `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_tomato_sauce_and_place_it_in_the_basket"}` |

## `molmoact2_libero_spatial`

- Route: MolmoAct2 Base + LIBERO Spatial
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/molmoact2_base_spatial.sh`
- Profile: `configs/molmoact2_libero.json` (`4c5063b381b5d48ff82865bfef73745fb97221797a1f313f00e010d4aa80c1c6`)
- Profile set: `libero_spatial`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [allenai/MolmoAct2-LIBERO](https://huggingface.co/allenai/MolmoAct2-LIBERO/tree/0d24a92bd1faf321ef497c3bbd5681af97c65aa2) at `0d24a92bd1faf321ef497c3bbd5681af97c65aa2`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `molmoact2_libero_spatial_canonical_50_per_task_v1`
- Standard route rows: 500
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/molmoact2_libero_spatial/benchmark_plan.json` (`bd73a90157c21636936c18a3935f1a3e9731ceb4cfe6fc20e05d4f96e3bd5b24`)
- Exact standard source: `manifests/benchmarks/molmoact2_libero_standard.json` (`965d82f2a695e567a3bb06946b3428769c4a6aaea12790b5c9048ad0eaf51527`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`
- Preset held-out tasks: `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`
- Preset sources: `manifests/episodes/molmoact2_libero_spatial_transfer.json` (`ce5738c89fda65d7fc655ad67b33d9aef3e9d74ae898e188592fdcccac70687f`)
- Preset evolution launch: `launch/routes/libero/molmoact2_base_spatial.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/molmoact2_base_spatial.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}` |

## `molmoact2_think_libero_goal`

- Route: MolmoAct2 Think + LIBERO Goal
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/molmoact2_think_goal.sh`
- Profile: `configs/molmoact2_think_libero_goal.json` (`543600f5a07685d80b0577b8ed36169b360a8843e7355b071229fb78620ba6ca`)
- Profile set: `libero_goal`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [allenai/MolmoAct2-Think-LIBERO](https://huggingface.co/allenai/MolmoAct2-Think-LIBERO/tree/593d25fcd3150e38eb05812fc3f9adb02927ec83) at `593d25fcd3150e38eb05812fc3f9adb02927ec83`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `molmoact2_libero_goal_canonical_50_per_task_v1`
- Standard route rows: 500
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/molmoact2_think_libero_goal/benchmark_plan.json` (`dc9c049b714ce067a2c92e81edd6a7f1dcc17811fca199ed6583ceecc3df2c7e`)
- Exact standard source: `manifests/benchmarks/molmoact2_think_libero_standard.json` (`db97900dc876df64fb5728bab51daf7b5d615a067a4a749f70bb0f20e7c5a3a9`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `open_the_middle_drawer_of_the_cabinet`, `put_the_bowl_on_the_stove`, `put_the_bowl_on_top_of_the_cabinet`, `put_the_wine_bottle_on_top_of_the_cabinet`
- Preset held-out tasks: `open_the_top_drawer_and_put_the_bowl_inside`, `put_the_bowl_on_the_plate`, `put_the_wine_bottle_on_the_rack`
- Preset sources: `manifests/episodes/molmoact2_libero_goal_transfer.json` (`aacbb4bf7c5008bc05187534b570da02b0283bc8b3acf9ad2b6b636f564f705d`)
- Preset evolution launch: `launch/routes/libero/molmoact2_think_goal.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/molmoact2_think_goal.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `open_the_middle_drawer_of_the_cabinet` | 50 | 300 | `{"task_id":"open_the_middle_drawer_of_the_cabinet"}` |
| `open_the_top_drawer_and_put_the_bowl_inside` | 50 | 300 | `{"task_id":"open_the_top_drawer_and_put_the_bowl_inside"}` |
| `push_the_plate_to_the_front_of_the_stove` | 50 | 300 | `{"task_id":"push_the_plate_to_the_front_of_the_stove"}` |
| `put_the_bowl_on_the_plate` | 50 | 300 | `{"task_id":"put_the_bowl_on_the_plate"}` |
| `put_the_bowl_on_the_stove` | 50 | 300 | `{"task_id":"put_the_bowl_on_the_stove"}` |
| `put_the_bowl_on_top_of_the_cabinet` | 50 | 300 | `{"task_id":"put_the_bowl_on_top_of_the_cabinet"}` |
| `put_the_cream_cheese_in_the_bowl` | 50 | 300 | `{"task_id":"put_the_cream_cheese_in_the_bowl"}` |
| `put_the_wine_bottle_on_the_rack` | 50 | 300 | `{"task_id":"put_the_wine_bottle_on_the_rack"}` |
| `put_the_wine_bottle_on_top_of_the_cabinet` | 50 | 300 | `{"task_id":"put_the_wine_bottle_on_top_of_the_cabinet"}` |
| `turn_on_the_stove` | 50 | 300 | `{"task_id":"turn_on_the_stove"}` |

## `molmoact2_think_libero_long`

- Route: MolmoAct2 Think + LIBERO Long
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/molmoact2_think_long.sh`
- Profile: `configs/molmoact2_think_libero_long.json` (`9f24402ca9c4715254e90a6770e039be1354afd9485339954ce5a5c302b9c8df`)
- Profile set: `libero_10`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [allenai/MolmoAct2-Think-LIBERO](https://huggingface.co/allenai/MolmoAct2-Think-LIBERO/tree/593d25fcd3150e38eb05812fc3f9adb02927ec83) at `593d25fcd3150e38eb05812fc3f9adb02927ec83`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `molmoact2_libero_10_canonical_50_per_task_v1`
- Standard route rows: 500
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/molmoact2_think_libero_long/benchmark_plan.json` (`11db213706847c9e92aaf1b791e299b309f6767ce93dbd086364a46e299dc6c5`)
- Exact standard source: `manifests/benchmarks/molmoact2_think_libero_standard.json` (`db97900dc876df64fb5728bab51daf7b5d615a067a4a749f70bb0f20e7c5a3a9`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`, `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`, `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`, `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
- Preset held-out tasks: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`, `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`
- Preset sources: `manifests/episodes/molmoact2_libero_long_transfer.json` (`f417482fcab976342f907ae436440dc520fd613a632adc5f1940f2bc3ed8841b`)
- Preset evolution launch: `launch/routes/libero/molmoact2_think_long.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/molmoact2_think_long.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 50 | 520 | `{"task_id":"KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}` |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 50 | 520 | `{"task_id":"KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}` |
| `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 50 | 520 | `{"task_id":"KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"}` |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 50 | 520 | `{"task_id":"KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}` |
| `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"}` |
| `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"}` |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 50 | 520 | `{"task_id":"LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"}` |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 50 | 520 | `{"task_id":"STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"}` |

## `molmoact2_think_libero_object`

- Route: MolmoAct2 Think + LIBERO Object
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/molmoact2_think_object.sh`
- Profile: `configs/molmoact2_think_libero_object.json` (`9499ad9032ffde290e65697abbeabc79a06a8e277e96f1bb797496a72af7d89f`)
- Profile set: `libero_object`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [allenai/MolmoAct2-Think-LIBERO](https://huggingface.co/allenai/MolmoAct2-Think-LIBERO/tree/593d25fcd3150e38eb05812fc3f9adb02927ec83) at `593d25fcd3150e38eb05812fc3f9adb02927ec83`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `molmoact2_libero_object_canonical_50_per_task_v1`
- Standard route rows: 500
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/molmoact2_think_libero_object/benchmark_plan.json` (`13178fc3fbb1826e01fe7c185d0c1d71b3b0f0c8ad943d4c4352e2f0df3e543f`)
- Exact standard source: `manifests/benchmarks/molmoact2_think_libero_standard.json` (`db97900dc876df64fb5728bab51daf7b5d615a067a4a749f70bb0f20e7c5a3a9`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`, `pick_up_the_bbq_sauce_and_place_it_in_the_basket`, `pick_up_the_cream_cheese_and_place_it_in_the_basket`, `pick_up_the_ketchup_and_place_it_in_the_basket`, `pick_up_the_salad_dressing_and_place_it_in_the_basket`
- Preset held-out tasks: `pick_up_the_butter_and_place_it_in_the_basket`, `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, `pick_up_the_milk_and_place_it_in_the_basket`, `pick_up_the_orange_juice_and_place_it_in_the_basket`, `pick_up_the_tomato_sauce_and_place_it_in_the_basket`
- Preset sources: `manifests/episodes/molmoact2_libero_object_transfer.json` (`89310c3093f85f1cb6fe5c2d33523dc3a4fdbcf57755abb6b713ebb90fcb6f48`)
- Preset evolution launch: `launch/routes/libero/molmoact2_think_object.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/molmoact2_think_object.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_alphabet_soup_and_place_it_in_the_basket"}` |
| `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_bbq_sauce_and_place_it_in_the_basket"}` |
| `pick_up_the_butter_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_butter_and_place_it_in_the_basket"}` |
| `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_chocolate_pudding_and_place_it_in_the_basket"}` |
| `pick_up_the_cream_cheese_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_cream_cheese_and_place_it_in_the_basket"}` |
| `pick_up_the_ketchup_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_ketchup_and_place_it_in_the_basket"}` |
| `pick_up_the_milk_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_milk_and_place_it_in_the_basket"}` |
| `pick_up_the_orange_juice_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_orange_juice_and_place_it_in_the_basket"}` |
| `pick_up_the_salad_dressing_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_salad_dressing_and_place_it_in_the_basket"}` |
| `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 50 | 280 | `{"task_id":"pick_up_the_tomato_sauce_and_place_it_in_the_basket"}` |

## `molmoact2_think_libero_spatial`

- Route: MolmoAct2 Think + LIBERO Spatial
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/molmoact2_think_spatial.sh`
- Profile: `configs/molmoact2_think_libero.json` (`89f585eaf2480edd6bcd86fff7e6efc507f9459e07fcdf1ed7bbbbacdb2ed907`)
- Profile set: `libero_spatial`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [allenai/MolmoAct2-Think-LIBERO](https://huggingface.co/allenai/MolmoAct2-Think-LIBERO/tree/593d25fcd3150e38eb05812fc3f9adb02927ec83) at `593d25fcd3150e38eb05812fc3f9adb02927ec83`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `molmoact2_libero_spatial_canonical_50_per_task_v1`
- Standard route rows: 500
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/molmoact2_think_libero_spatial/benchmark_plan.json` (`f8747a83d661bb21f6ff608e07db1d0b02a575fec3effb9d8b3e141e30a5160d`)
- Exact standard source: `manifests/benchmarks/molmoact2_think_libero_standard.json` (`db97900dc876df64fb5728bab51daf7b5d615a067a4a749f70bb0f20e7c5a3a9`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`
- Preset held-out tasks: `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`
- Preset sources: `manifests/episodes/molmoact2_libero_spatial_transfer.json` (`ce5738c89fda65d7fc655ad67b33d9aef3e9d74ae898e188592fdcccac70687f`)
- Preset evolution launch: `launch/routes/libero/molmoact2_think_spatial.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/molmoact2_think_spatial.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 50 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}` |

## `openvla_simpler_google_va`

- Route: OpenVLA base + SimplerEnv Google Variant Aggregation
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/simpler/openvla_google_va.sh`
- Profile: `configs/openvla_simpler_google_va.json` (`83e18a43f159fb1489c5744cb43fc8821ec26133bb017a13b2b80335edc8c79d`)
- Profile set: `simpler_google_va`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b/tree/47a0ec7fc4ec123775a391911046cf33cf9ed83f) at `47a0ec7fc4ec123775a391911046cf33cf9ed83f`
- Full benchmark status: `ready`
- Metric: `task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `openvla_google_va_complete_grid_v1`
- Standard route rows: 1992
- Comparability: This complete project grid is not an OpenVLA paper-headline protocol.
- Route benchmark plan: `manifests/benchmarks/openvla_simpler_google_va_complete_grid_v1.json` (`267a455a7631f3d0071fc9b7ea97d8ac72f7a921e25d77afc6b367e1a36de58e`)
- Exact standard source: `manifests/benchmarks/openvla_simpler_google_va_complete_grid_v1.json` (`267a455a7631f3d0071fc9b7ea97d8ac72f7a921e25d77afc6b367e1a36de58e`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `google_robot_open_drawer`
- Preset held-out tasks: `google_robot_close_drawer`
- Preset sources: `manifests/episodes/openvla_simpler_google_va_drawer_transfer_v1.json` (`f8386d679bcd69a9fdfdb206f45bfe590b1ae11dc1cd0af3be75b19cfae9e84c`)
- Preset evolution launch: `launch/routes/simpler/openvla_google_va.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/simpler/openvla_google_va.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This SimplerEnv route has no calibrated metric-depth observation and no controller that executes GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `google_robot_close_drawer` | 189 | 113 | `{"task_id":"google_robot_close_drawer"}` |
| `google_robot_move_near` | 600 | 80 | `{"task_id":"google_robot_move_near"}` |
| `google_robot_open_drawer` | 189 | 113 | `{"task_id":"google_robot_open_drawer"}` |
| `google_robot_pick_coke_can` | 825 | 80 | `{"task_id":"google_robot_pick_coke_can"}` |
| `google_robot_place_apple_in_closed_top_drawer` | 189 | 200 | `{"task_id":"google_robot_place_apple_in_closed_top_drawer"}` |

## `openvla_simpler_google_vm`

- Route: OpenVLA base + SimplerEnv Google Visual Matching
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/simpler/openvla_google_vm.sh`
- Profile: `configs/openvla_simpler_google_vm.json` (`e96c7e1b36acb41ae8f04a9b3c35234a4998270b1c1c13df9f4e86a17fc75d97`)
- Profile set: `simpler_google_vm`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b/tree/47a0ec7fc4ec123775a391911046cf33cf9ed83f) at `47a0ec7fc4ec123775a391911046cf33cf9ed83f`
- Full benchmark status: `ready`
- Metric: `task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `openvla_google_vm_complete_grid_v1`
- Standard route rows: 864
- Comparability: This complete project grid is not an OpenVLA paper-headline protocol.
- Route benchmark plan: `manifests/benchmarks/openvla_simpler_google_vm_complete_grid_v1.json` (`af350b456a7dd50f2df9783eee4daa53c2d32793e1c3895052423fbe1e41d976`)
- Exact standard source: `manifests/benchmarks/openvla_simpler_google_vm_complete_grid_v1.json` (`af350b456a7dd50f2df9783eee4daa53c2d32793e1c3895052423fbe1e41d976`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `google_robot_open_drawer`
- Preset held-out tasks: `google_robot_close_drawer`
- Preset sources: `manifests/episodes/openvla_simpler_google_vm_drawer_transfer_v1.json` (`f49a04c20b39f4960f3bef5a37c878b3c5f628ea987c68868cfef9d1a25a4755`)
- Preset evolution launch: `launch/routes/simpler/openvla_google_vm.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/simpler/openvla_google_vm.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This SimplerEnv route has no calibrated metric-depth observation and no controller that executes GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `google_robot_close_drawer` | 108 | 113 | `{"task_id":"google_robot_close_drawer"}` |
| `google_robot_move_near` | 240 | 80 | `{"task_id":"google_robot_move_near"}` |
| `google_robot_open_drawer` | 108 | 113 | `{"task_id":"google_robot_open_drawer"}` |
| `google_robot_pick_coke_can` | 300 | 80 | `{"task_id":"google_robot_pick_coke_can"}` |
| `google_robot_place_apple_in_closed_top_drawer` | 108 | 200 | `{"task_id":"google_robot_place_apple_in_closed_top_drawer"}` |

## `pi05_libero_goal`

- Route: LeRobot pi0.5 + LIBERO Goal
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/lerobot_pi05_goal.sh`
- Profile: `configs/pi05_libero_goal.json` (`bf9ea014672c20067c24ba52b08b4a53e494f062d4b8d5ca54af7895b1f7b85d`)
- Profile set: `libero_goal`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [lerobot/pi05_libero_finetuned](https://huggingface.co/lerobot/pi05_libero_finetuned/tree/dbf8a3f794a9c4297b44f40b752712f50073d945) at `dbf8a3f794a9c4297b44f40b752712f50073d945`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `pi05_lerobot_libero_goal_canonical_10_per_task_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/pi05_libero_goal/benchmark_plan.json` (`234b060b37c38afa888c78d057d33543cc1b74f130ee9ba8c57e1f58e738c837`)
- Exact standard source: `manifests/benchmarks/pi05_libero_standard.json` (`d05e9572a9f5f13b545c9a8cea2e3de4bd9adf09347065c76a65184837e62d72`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `open_the_middle_drawer_of_the_cabinet`, `put_the_bowl_on_the_stove`, `put_the_bowl_on_top_of_the_cabinet`, `put_the_wine_bottle_on_top_of_the_cabinet`
- Preset held-out tasks: `open_the_top_drawer_and_put_the_bowl_inside`, `put_the_bowl_on_the_plate`, `put_the_wine_bottle_on_the_rack`
- Preset sources: `manifests/episodes/pi05_libero_goal_transfer.json` (`30372ad39420899d9032fb864da058529340335d88b9663ee97cbb7210bbe846`)
- Preset evolution launch: `launch/routes/libero/lerobot_pi05_goal.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/lerobot_pi05_goal.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `open_the_middle_drawer_of_the_cabinet` | 10 | 300 | `{"task_id":"open_the_middle_drawer_of_the_cabinet"}` |
| `open_the_top_drawer_and_put_the_bowl_inside` | 10 | 300 | `{"task_id":"open_the_top_drawer_and_put_the_bowl_inside"}` |
| `push_the_plate_to_the_front_of_the_stove` | 10 | 300 | `{"task_id":"push_the_plate_to_the_front_of_the_stove"}` |
| `put_the_bowl_on_the_plate` | 10 | 300 | `{"task_id":"put_the_bowl_on_the_plate"}` |
| `put_the_bowl_on_the_stove` | 10 | 300 | `{"task_id":"put_the_bowl_on_the_stove"}` |
| `put_the_bowl_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"put_the_bowl_on_top_of_the_cabinet"}` |
| `put_the_cream_cheese_in_the_bowl` | 10 | 300 | `{"task_id":"put_the_cream_cheese_in_the_bowl"}` |
| `put_the_wine_bottle_on_the_rack` | 10 | 300 | `{"task_id":"put_the_wine_bottle_on_the_rack"}` |
| `put_the_wine_bottle_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"put_the_wine_bottle_on_top_of_the_cabinet"}` |
| `turn_on_the_stove` | 10 | 300 | `{"task_id":"turn_on_the_stove"}` |

## `pi05_libero_long`

- Route: LeRobot pi0.5 + LIBERO Long
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/lerobot_pi05_long.sh`
- Profile: `configs/pi05_libero_long.json` (`b645c203bf947b72cc91c4eab41cbc07a484c3b95e435cd1a4155d80c169e4b4`)
- Profile set: `libero_10`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [lerobot/pi05_libero_finetuned](https://huggingface.co/lerobot/pi05_libero_finetuned/tree/dbf8a3f794a9c4297b44f40b752712f50073d945) at `dbf8a3f794a9c4297b44f40b752712f50073d945`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `pi05_lerobot_libero_10_canonical_10_per_task_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/pi05_libero_long/benchmark_plan.json` (`40b89a78389cbba80d9e7a21178a9f271014270349b651300c0baddbd8c48265`)
- Exact standard source: `manifests/benchmarks/pi05_libero_standard.json` (`d05e9572a9f5f13b545c9a8cea2e3de4bd9adf09347065c76a65184837e62d72`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`, `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`, `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`, `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
- Preset held-out tasks: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`, `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`
- Preset sources: `manifests/episodes/pi05_libero_long_transfer.json` (`3500b3f11ec445bb36a40f34fcd7c048571a9bc1cad9a2c5b84f9393a594b347`)
- Preset evolution launch: `launch/routes/libero/lerobot_pi05_long.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/lerobot_pi05_long.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 10 | 520 | `{"task_id":"KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}` |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 10 | 520 | `{"task_id":"KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}` |
| `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 10 | 520 | `{"task_id":"KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"}` |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 10 | 520 | `{"task_id":"KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}` |
| `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"}` |
| `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"}` |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"}` |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 10 | 520 | `{"task_id":"STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"}` |

## `pi05_libero_object`

- Route: LeRobot pi0.5 + LIBERO Object
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/lerobot_pi05_object.sh`
- Profile: `configs/pi05_libero_object.json` (`b17420ebf3823e9b896a6f502df3cb8ea76e5b77bd4f6c6d8207ca8c57753468`)
- Profile set: `libero_object`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [lerobot/pi05_libero_finetuned](https://huggingface.co/lerobot/pi05_libero_finetuned/tree/dbf8a3f794a9c4297b44f40b752712f50073d945) at `dbf8a3f794a9c4297b44f40b752712f50073d945`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `pi05_lerobot_libero_object_canonical_10_per_task_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/pi05_libero_object/benchmark_plan.json` (`5985565a781e63f0ea595cd879e8ff4ac234697d686363287402c951700ab71a`)
- Exact standard source: `manifests/benchmarks/pi05_libero_standard.json` (`d05e9572a9f5f13b545c9a8cea2e3de4bd9adf09347065c76a65184837e62d72`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`, `pick_up_the_bbq_sauce_and_place_it_in_the_basket`, `pick_up_the_cream_cheese_and_place_it_in_the_basket`, `pick_up_the_ketchup_and_place_it_in_the_basket`, `pick_up_the_salad_dressing_and_place_it_in_the_basket`
- Preset held-out tasks: `pick_up_the_butter_and_place_it_in_the_basket`, `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, `pick_up_the_milk_and_place_it_in_the_basket`, `pick_up_the_orange_juice_and_place_it_in_the_basket`, `pick_up_the_tomato_sauce_and_place_it_in_the_basket`
- Preset sources: `manifests/episodes/pi05_libero_object_transfer.json` (`d6638ca261f40008574b1164cd98ea1edd53e75e33b8d501fd37bf1fcd40559f`)
- Preset evolution launch: `launch/routes/libero/lerobot_pi05_object.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/lerobot_pi05_object.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_alphabet_soup_and_place_it_in_the_basket"}` |
| `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_bbq_sauce_and_place_it_in_the_basket"}` |
| `pick_up_the_butter_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_butter_and_place_it_in_the_basket"}` |
| `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_chocolate_pudding_and_place_it_in_the_basket"}` |
| `pick_up_the_cream_cheese_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_cream_cheese_and_place_it_in_the_basket"}` |
| `pick_up_the_ketchup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_ketchup_and_place_it_in_the_basket"}` |
| `pick_up_the_milk_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_milk_and_place_it_in_the_basket"}` |
| `pick_up_the_orange_juice_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_orange_juice_and_place_it_in_the_basket"}` |
| `pick_up_the_salad_dressing_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_salad_dressing_and_place_it_in_the_basket"}` |
| `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_tomato_sauce_and_place_it_in_the_basket"}` |

## `pi05_libero_spatial`

- Route: LeRobot pi0.5 + LIBERO Spatial
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/lerobot_pi05_spatial.sh`
- Profile: `configs/pi05_libero.json` (`6d0357e2f84bbb4d9248551fb8549dd091f4d78e24f859f8756362e0b08f0a54`)
- Profile set: `libero_spatial`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [lerobot/pi05_libero_finetuned](https://huggingface.co/lerobot/pi05_libero_finetuned/tree/dbf8a3f794a9c4297b44f40b752712f50073d945) at `dbf8a3f794a9c4297b44f40b752712f50073d945`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `pi05_lerobot_libero_spatial_canonical_10_per_task_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/pi05_libero_spatial/benchmark_plan.json` (`1a44e3e4679b96a3844c370e8aaaa65c07ddc6ce962033c6f5bd1c730fa06cd2`)
- Exact standard source: `manifests/benchmarks/pi05_libero_standard.json` (`d05e9572a9f5f13b545c9a8cea2e3de4bd9adf09347065c76a65184837e62d72`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`
- Preset held-out tasks: `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`
- Preset sources: `manifests/episodes/pi05_libero_spatial_transfer.json` (`5cda1aa7c207cbda2303ed228e59c2d164b9ba5957de75e721f3ddf31fb76c6a`)
- Preset evolution launch: `launch/routes/libero/lerobot_pi05_spatial.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/lerobot_pi05_spatial.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 280 | `{"task_id":"pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}` |

## `rldx_robocasa365`

- Route: RLDX-1 + RoboCasa365 Target-50
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/robocasa365/rldx_target50.sh`
- Profile: `configs/rldx_robocasa365.json` (`2cd39e84b37607d1b853b42159b39345a7b40da4b51092ba3b24a594d1ed0960`)
- Profile set: `robocasa365_target`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLWRLD/RLDX-1-FT-RC365](https://huggingface.co/RLWRLD/RLDX-1-FT-RC365/tree/587e9ecdcc5e7184fcc17f58713908edff5af041) at `587e9ecdcc5e7184fcc17f58713908edff5af041`
- Full benchmark status: `ready`
- Metric: `equal_group_task_macro_success`
- Default resources: 2 GPUs, 2 workers per GPU, 4 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rldx_robocasa365_target50_public_50_per_task_v1`
- Standard route rows: 2500
- Comparability: The task and trial schedule matches the public RLDX Target-50 protocol; the evolved agent adds frozen tools and is not Harness-VLA.
- Route benchmark plan: `manifests/benchmarks/rldx_robocasa365_target50_public_50_per_task_v1.json` (`6453f97e2888c029151d4d7571b62d23636788a9f46d912b2d0b7bd6148f7b26`)
- Exact standard source: `manifests/benchmarks/rldx_robocasa365_target50_public_50_per_task_v1.json` (`6453f97e2888c029151d4d7571b62d23636788a9f46d912b2d0b7bd6148f7b26`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `OpenCabinet`, `OpenDrawer`, `PickPlaceCounterToCabinet`, `TurnOnMicrowave`
- Preset held-out tasks: `CloseFridge`, `PickPlaceCounterToStove`, `SlideDishwasherRack`, `TurnOnElectricKettle`
- Preset sources: `manifests/episodes/rldx_robocasa365_target_transfer.json` (`3ea35698c590af295d877b6aa8854d1faa7b182cdbb5a0739a83e47d277b9c92`)
- Preset evolution launch: `launch/routes/robocasa365/rldx_target50.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/robocasa365/rldx_target50.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | The public RLDX RoboCasa365 route exposes RGB without metric depth or camera calibration, so GraspGen poses cannot be grounded or executed safely. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `ArrangeBreadBasket` | 50 | 2900 | `{"task_id":"ArrangeBreadBasket"}` |
| `ArrangeTea` | 50 | 1500 | `{"task_id":"ArrangeTea"}` |
| `BreadSelection` | 50 | 1300 | `{"task_id":"BreadSelection"}` |
| `CategorizeCondiments` | 50 | 1100 | `{"task_id":"CategorizeCondiments"}` |
| `CloseBlenderLid` | 50 | 600 | `{"task_id":"CloseBlenderLid"}` |
| `CloseFridge` | 50 | 600 | `{"task_id":"CloseFridge"}` |
| `CloseToasterOvenDoor` | 50 | 300 | `{"task_id":"CloseToasterOvenDoor"}` |
| `CoffeeSetupMug` | 50 | 400 | `{"task_id":"CoffeeSetupMug"}` |
| `CuttingToolSelection` | 50 | 800 | `{"task_id":"CuttingToolSelection"}` |
| `DeliverStraw` | 50 | 1700 | `{"task_id":"DeliverStraw"}` |
| `GarnishPancake` | 50 | 1800 | `{"task_id":"GarnishPancake"}` |
| `GatherTableware` | 50 | 1500 | `{"task_id":"GatherTableware"}` |
| `GetToastedBread` | 50 | 2000 | `{"task_id":"GetToastedBread"}` |
| `HeatKebabSandwich` | 50 | 1800 | `{"task_id":"HeatKebabSandwich"}` |
| `KettleBoiling` | 50 | 1000 | `{"task_id":"KettleBoiling"}` |
| `LoadDishwasher` | 50 | 1200 | `{"task_id":"LoadDishwasher"}` |
| `MakeIceLemonade` | 50 | 2000 | `{"task_id":"MakeIceLemonade"}` |
| `NavigateKitchen` | 50 | 300 | `{"task_id":"NavigateKitchen"}` |
| `OpenCabinet` | 50 | 700 | `{"task_id":"OpenCabinet"}` |
| `OpenDrawer` | 50 | 500 | `{"task_id":"OpenDrawer"}` |
| `OpenStandMixerHead` | 50 | 300 | `{"task_id":"OpenStandMixerHead"}` |
| `PackIdenticalLunches` | 50 | 2600 | `{"task_id":"PackIdenticalLunches"}` |
| `PanTransfer` | 50 | 1200 | `{"task_id":"PanTransfer"}` |
| `PickPlaceCounterToCabinet` | 50 | 500 | `{"task_id":"PickPlaceCounterToCabinet"}` |
| `PickPlaceCounterToStove` | 50 | 400 | `{"task_id":"PickPlaceCounterToStove"}` |
| `PickPlaceDrawerToCounter` | 50 | 500 | `{"task_id":"PickPlaceDrawerToCounter"}` |
| `PickPlaceSinkToCounter` | 50 | 600 | `{"task_id":"PickPlaceSinkToCounter"}` |
| `PickPlaceToasterToCounter` | 50 | 400 | `{"task_id":"PickPlaceToasterToCounter"}` |
| `PortionHotDogs` | 50 | 1500 | `{"task_id":"PortionHotDogs"}` |
| `PreSoakPan` | 50 | 1600 | `{"task_id":"PreSoakPan"}` |
| `PrepareCoffee` | 50 | 1200 | `{"task_id":"PrepareCoffee"}` |
| `RecycleBottlesByType` | 50 | 1900 | `{"task_id":"RecycleBottlesByType"}` |
| `RinseSinkBasin` | 50 | 900 | `{"task_id":"RinseSinkBasin"}` |
| `ScrubCuttingBoard` | 50 | 800 | `{"task_id":"ScrubCuttingBoard"}` |
| `SearingMeat` | 50 | 2900 | `{"task_id":"SearingMeat"}` |
| `SeparateFreezerRack` | 50 | 1600 | `{"task_id":"SeparateFreezerRack"}` |
| `SetUpCuttingStation` | 50 | 1600 | `{"task_id":"SetUpCuttingStation"}` |
| `SlideDishwasherRack` | 50 | 300 | `{"task_id":"SlideDishwasherRack"}` |
| `StackBowlsCabinet` | 50 | 1400 | `{"task_id":"StackBowlsCabinet"}` |
| `SteamInMicrowave` | 50 | 1400 | `{"task_id":"SteamInMicrowave"}` |
| `StirVegetables` | 50 | 1600 | `{"task_id":"StirVegetables"}` |
| `StoreLeftoversInBowl` | 50 | 1700 | `{"task_id":"StoreLeftoversInBowl"}` |
| `TurnOffStove` | 50 | 500 | `{"task_id":"TurnOffStove"}` |
| `TurnOnElectricKettle` | 50 | 300 | `{"task_id":"TurnOnElectricKettle"}` |
| `TurnOnMicrowave` | 50 | 300 | `{"task_id":"TurnOnMicrowave"}` |
| `TurnOnSinkFaucet` | 50 | 400 | `{"task_id":"TurnOnSinkFaucet"}` |
| `WaffleReheat` | 50 | 2700 | `{"task_id":"WaffleReheat"}` |
| `WashFruitColander` | 50 | 2100 | `{"task_id":"WashFruitColander"}` |
| `WashLettuce` | 50 | 1100 | `{"task_id":"WashLettuce"}` |
| `WeighIngredients` | 50 | 2000 | `{"task_id":"WeighIngredients"}` |

## `rlinf_pi05_libero_goal`

- Route: RLinf pi0.5 + LIBERO Goal
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/rlinf_pi05_goal.sh`
- Profile: `configs/rlinf_pi05_libero_goal.json` (`0cca10a4550679bb780055966421ada6b4f1c23de33bb1c7863e37942b2b191b`)
- Profile set: `libero_goal`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT/tree/6222623f635769bfc73c9472e29fab9b7fd8e027) at `6222623f635769bfc73c9472e29fab9b7fd8e027`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rlinf_pi05_libero_goal_canonical_10_per_task_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/rlinf_pi05_libero_goal/benchmark_plan.json` (`f4f4e1282e4f65698d213851c8ae4b829977f335001ffffb3ac4da4aa592b898`)
- Exact standard source: `manifests/benchmarks/rlinf_pi05_libero_standard.json` (`ebf9966972d174408d6563e380b82d6d7c3b2438723d8f459a733c1c3cad3e55`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `open_the_middle_drawer_of_the_cabinet`, `put_the_bowl_on_the_stove`, `put_the_bowl_on_top_of_the_cabinet`, `put_the_wine_bottle_on_top_of_the_cabinet`
- Preset held-out tasks: `open_the_top_drawer_and_put_the_bowl_inside`, `put_the_bowl_on_the_plate`, `put_the_wine_bottle_on_the_rack`
- Preset sources: `manifests/episodes/rlinf_pi05_libero_goal_related_transfer.json` (`a8fc8d2e9d1bde32022492ebe7d13afbe24be4e64b2c31a09f7ac3bfe07de3a9`)
- Preset evolution launch: `launch/routes/libero/rlinf_pi05_goal.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/rlinf_pi05_goal.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `open_the_middle_drawer_of_the_cabinet` | 10 | 300 | `{"task_id":"open_the_middle_drawer_of_the_cabinet"}` |
| `open_the_top_drawer_and_put_the_bowl_inside` | 10 | 300 | `{"task_id":"open_the_top_drawer_and_put_the_bowl_inside"}` |
| `push_the_plate_to_the_front_of_the_stove` | 10 | 300 | `{"task_id":"push_the_plate_to_the_front_of_the_stove"}` |
| `put_the_bowl_on_the_plate` | 10 | 300 | `{"task_id":"put_the_bowl_on_the_plate"}` |
| `put_the_bowl_on_the_stove` | 10 | 300 | `{"task_id":"put_the_bowl_on_the_stove"}` |
| `put_the_bowl_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"put_the_bowl_on_top_of_the_cabinet"}` |
| `put_the_cream_cheese_in_the_bowl` | 10 | 300 | `{"task_id":"put_the_cream_cheese_in_the_bowl"}` |
| `put_the_wine_bottle_on_the_rack` | 10 | 300 | `{"task_id":"put_the_wine_bottle_on_the_rack"}` |
| `put_the_wine_bottle_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"put_the_wine_bottle_on_top_of_the_cabinet"}` |
| `turn_on_the_stove` | 10 | 300 | `{"task_id":"turn_on_the_stove"}` |

## `rlinf_pi05_libero_long`

- Route: RLinf pi0.5 + LIBERO Long
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/rlinf_pi05_long.sh`
- Profile: `configs/rlinf_pi05_libero_long.json` (`61985cf5730fad3064263aae062edc12af94106f2964ac8081fcf9b3b19b4ca4`)
- Profile set: `libero_10`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT/tree/6222623f635769bfc73c9472e29fab9b7fd8e027) at `6222623f635769bfc73c9472e29fab9b7fd8e027`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rlinf_pi05_libero_long_canonical_10_per_task_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/rlinf_pi05_libero_long/benchmark_plan.json` (`a0a6b2d92d59c418a82037090ab87847b4043b2bd0fcaa82588c775e847dd13a`)
- Exact standard source: `manifests/benchmarks/rlinf_pi05_libero_standard.json` (`ebf9966972d174408d6563e380b82d6d7c3b2438723d8f459a733c1c3cad3e55`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`, `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`, `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`, `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
- Preset held-out tasks: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`, `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`
- Preset sources: `manifests/episodes/rlinf_pi05_libero_long_related_transfer.json` (`031bbe1037d8ca788028bb37a57302a8b58a18a332f38b76b86d51a35f7acda9`)
- Preset evolution launch: `launch/routes/libero/rlinf_pi05_long.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/rlinf_pi05_long.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 10 | 520 | `{"task_id":"KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}` |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 10 | 520 | `{"task_id":"KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}` |
| `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 10 | 520 | `{"task_id":"KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"}` |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 10 | 520 | `{"task_id":"KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}` |
| `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"}` |
| `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"}` |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 10 | 520 | `{"task_id":"LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"}` |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 10 | 520 | `{"task_id":"STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"}` |

## `rlinf_pi05_libero_object`

- Route: RLinf pi0.5 + LIBERO Object
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/rlinf_pi05_object.sh`
- Profile: `configs/rlinf_pi05_libero_object.json` (`dcbe9ea8a7b0611e067cb4749102887c725063819b28ce58959d0ec2324fdf6f`)
- Profile set: `libero_object`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT/tree/6222623f635769bfc73c9472e29fab9b7fd8e027) at `6222623f635769bfc73c9472e29fab9b7fd8e027`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rlinf_pi05_libero_object_canonical_10_per_task_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/rlinf_pi05_libero_object/benchmark_plan.json` (`131946dbee06c83659bfbb28b3a9a7cc373f7f7f498d7d3c58ea878f3e2bef12`)
- Exact standard source: `manifests/benchmarks/rlinf_pi05_libero_standard.json` (`ebf9966972d174408d6563e380b82d6d7c3b2438723d8f459a733c1c3cad3e55`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`, `pick_up_the_bbq_sauce_and_place_it_in_the_basket`, `pick_up_the_cream_cheese_and_place_it_in_the_basket`, `pick_up_the_ketchup_and_place_it_in_the_basket`, `pick_up_the_salad_dressing_and_place_it_in_the_basket`
- Preset held-out tasks: `pick_up_the_butter_and_place_it_in_the_basket`, `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, `pick_up_the_milk_and_place_it_in_the_basket`, `pick_up_the_orange_juice_and_place_it_in_the_basket`, `pick_up_the_tomato_sauce_and_place_it_in_the_basket`
- Preset sources: `manifests/episodes/rlinf_pi05_libero_object_related_transfer.json` (`f0317a82779d3f6793d4039b196e964605d07c4d037969cf75141a853568280b`)
- Preset evolution launch: `launch/routes/libero/rlinf_pi05_object.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/rlinf_pi05_object.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_alphabet_soup_and_place_it_in_the_basket"}` |
| `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_bbq_sauce_and_place_it_in_the_basket"}` |
| `pick_up_the_butter_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_butter_and_place_it_in_the_basket"}` |
| `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_chocolate_pudding_and_place_it_in_the_basket"}` |
| `pick_up_the_cream_cheese_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_cream_cheese_and_place_it_in_the_basket"}` |
| `pick_up_the_ketchup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_ketchup_and_place_it_in_the_basket"}` |
| `pick_up_the_milk_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_milk_and_place_it_in_the_basket"}` |
| `pick_up_the_orange_juice_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_orange_juice_and_place_it_in_the_basket"}` |
| `pick_up_the_salad_dressing_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_salad_dressing_and_place_it_in_the_basket"}` |
| `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"pick_up_the_tomato_sauce_and_place_it_in_the_basket"}` |

## `rlinf_pi05_libero_pro_10_swap`

- Route: RLinf pi0.5 + LIBERO-Pro 10 swap
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero_pro/rlinf_pi05_10_swap.sh`
- Profile: `configs/rlinf_pi05_libero_pro_10_swap.json` (`46eaa9bff58dbb82066708606096a8a2b57f523a1f36c9bc765ece6e6ba033bd`)
- Profile set: `libero_pro_10_swap`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT/tree/6222623f635769bfc73c9472e29fab9b7fd8e027) at `6222623f635769bfc73c9472e29fab9b7fd8e027`
- Full benchmark status: `ready`
- Metric: `equal_cell_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rlinf_pi05_libero_pro_10_swap_paper_v3_10_seed_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task LIBERO-Pro cell, not the eight-cell headline. It uses the released RLinf policy without the unreleased Harness memory agent.
- Route benchmark plan: `routes/libero_pro/rlinf_pi05_libero_pro_10_swap/benchmark_plan.json` (`d61dc2289f316fcd94deb71a5604887590be38a9f18fd6cd18b1beb4e6c7e8f8`)
- Exact standard source: `manifests/benchmarks/rlinf_pi05_libero_pro_harness_paper_v3.json` (`03e6adde51c602740f3bf5c9f3d0e55640458ffe5d88d472c70ba730f78f0412`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `libero_pro_10_swap::KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `libero_pro_10_swap::KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`, `libero_pro_10_swap::LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`, `libero_pro_10_swap::LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`, `libero_pro_10_swap::STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`
- Preset held-out tasks: `libero_pro_10_swap::KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`, `libero_pro_10_swap::KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, `libero_pro_10_swap::LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, `libero_pro_10_swap::LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`, `libero_pro_10_swap::LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`
- Preset sources: `manifests/episodes/rlinf_pi05_libero_pro_10_swap_related_transfer.json` (`ea7985e243569c264752e3d823226df6d7104880c6b3c94f2c9d918101d1c711`)
- Preset evolution launch: `launch/routes/libero_pro/rlinf_pi05_10_swap.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero_pro/rlinf_pi05_10_swap.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `libero_pro_10_swap::KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 10 | 520 | `{"task_id":"libero_pro_10_swap::KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}` |
| `libero_pro_10_swap::KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 10 | 520 | `{"task_id":"libero_pro_10_swap::KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}` |
| `libero_pro_10_swap::KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 10 | 520 | `{"task_id":"libero_pro_10_swap::KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"}` |
| `libero_pro_10_swap::KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 10 | 520 | `{"task_id":"libero_pro_10_swap::KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}` |
| `libero_pro_10_swap::LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 10 | 520 | `{"task_id":"libero_pro_10_swap::LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"}` |
| `libero_pro_10_swap::LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 10 | 520 | `{"task_id":"libero_pro_10_swap::LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"}` |
| `libero_pro_10_swap::LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 10 | 520 | `{"task_id":"libero_pro_10_swap::LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"}` |
| `libero_pro_10_swap::LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 10 | 520 | `{"task_id":"libero_pro_10_swap::LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"}` |
| `libero_pro_10_swap::LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 10 | 520 | `{"task_id":"libero_pro_10_swap::LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"}` |
| `libero_pro_10_swap::STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 10 | 520 | `{"task_id":"libero_pro_10_swap::STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"}` |

## `rlinf_pi05_libero_pro_10_task`

- Route: RLinf pi0.5 + LIBERO-Pro 10 task
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero_pro/rlinf_pi05_10_task.sh`
- Profile: `configs/rlinf_pi05_libero_pro_10_task.json` (`2c7f759676015d04f939985ae6933198e6f79dfaadab14ad4b81ebb368ceb9a6`)
- Profile set: `libero_pro_10_task`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT/tree/6222623f635769bfc73c9472e29fab9b7fd8e027) at `6222623f635769bfc73c9472e29fab9b7fd8e027`
- Full benchmark status: `ready`
- Metric: `equal_cell_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rlinf_pi05_libero_pro_10_task_paper_v3_10_seed_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task LIBERO-Pro cell, not the eight-cell headline. It uses the released RLinf policy without the unreleased Harness memory agent.
- Route benchmark plan: `routes/libero_pro/rlinf_pi05_libero_pro_10_task/benchmark_plan.json` (`8c726d9cb57780cffe098f6e288c81445f0ee931dc55654dba0bd3f075d197bf`)
- Exact standard source: `manifests/benchmarks/rlinf_pi05_libero_pro_harness_paper_v3.json` (`03e6adde51c602740f3bf5c9f3d0e55640458ffe5d88d472c70ba730f78f0412`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `libero_pro_10_task::KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `libero_pro_10_task::KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`, `libero_pro_10_task::LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`, `libero_pro_10_task::LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`, `libero_pro_10_task::STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`
- Preset held-out tasks: `libero_pro_10_task::KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`, `libero_pro_10_task::KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, `libero_pro_10_task::LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, `libero_pro_10_task::LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`, `libero_pro_10_task::LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`
- Preset sources: `manifests/episodes/rlinf_pi05_libero_pro_10_task_related_transfer.json` (`54ad0bdb3acb5420a051640a0c98383936a27a80ffc8db8ffc9a5a3714707319`)
- Preset evolution launch: `launch/routes/libero_pro/rlinf_pi05_10_task.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero_pro/rlinf_pi05_10_task.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `libero_pro_10_task::KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 10 | 520 | `{"task_id":"libero_pro_10_task::KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}` |
| `libero_pro_10_task::KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 10 | 520 | `{"task_id":"libero_pro_10_task::KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}` |
| `libero_pro_10_task::KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 10 | 520 | `{"task_id":"libero_pro_10_task::KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"}` |
| `libero_pro_10_task::KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 10 | 520 | `{"task_id":"libero_pro_10_task::KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}` |
| `libero_pro_10_task::LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 10 | 520 | `{"task_id":"libero_pro_10_task::LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"}` |
| `libero_pro_10_task::LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 10 | 520 | `{"task_id":"libero_pro_10_task::LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"}` |
| `libero_pro_10_task::LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 10 | 520 | `{"task_id":"libero_pro_10_task::LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"}` |
| `libero_pro_10_task::LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 10 | 520 | `{"task_id":"libero_pro_10_task::LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"}` |
| `libero_pro_10_task::LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 10 | 520 | `{"task_id":"libero_pro_10_task::LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"}` |
| `libero_pro_10_task::STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 10 | 520 | `{"task_id":"libero_pro_10_task::STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"}` |

## `rlinf_pi05_libero_pro_goal_swap`

- Route: RLinf pi0.5 + LIBERO-Pro goal swap
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero_pro/rlinf_pi05_goal_swap.sh`
- Profile: `configs/rlinf_pi05_libero_pro_goal_swap.json` (`b4362ed83e7112ab9f375f97dbc868a9e6f58c6dc5796c226a00974e328a2a3c`)
- Profile set: `libero_pro_goal_swap`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT/tree/6222623f635769bfc73c9472e29fab9b7fd8e027) at `6222623f635769bfc73c9472e29fab9b7fd8e027`
- Full benchmark status: `ready`
- Metric: `equal_cell_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rlinf_pi05_libero_pro_goal_swap_paper_v3_10_seed_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task LIBERO-Pro cell, not the eight-cell headline. It uses the released RLinf policy without the unreleased Harness memory agent.
- Route benchmark plan: `routes/libero_pro/rlinf_pi05_libero_pro_goal_swap/benchmark_plan.json` (`97eb17e3e24902abfd69ab3a804e5b02ab0127ce85dd8d789048f9f65bbf4fd0`)
- Exact standard source: `manifests/benchmarks/rlinf_pi05_libero_pro_harness_paper_v3.json` (`03e6adde51c602740f3bf5c9f3d0e55640458ffe5d88d472c70ba730f78f0412`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `libero_pro_goal_swap::open_the_middle_drawer_of_the_cabinet`, `libero_pro_goal_swap::open_the_top_drawer_and_put_the_bowl_inside`, `libero_pro_goal_swap::put_the_bowl_on_the_stove`, `libero_pro_goal_swap::put_the_wine_bottle_on_top_of_the_cabinet`, `libero_pro_goal_swap::turn_on_the_stove`
- Preset held-out tasks: `libero_pro_goal_swap::push_the_plate_to_the_front_of_the_stove`, `libero_pro_goal_swap::put_the_bowl_on_the_plate`, `libero_pro_goal_swap::put_the_bowl_on_top_of_the_cabinet`, `libero_pro_goal_swap::put_the_cream_cheese_in_the_bowl`, `libero_pro_goal_swap::put_the_wine_bottle_on_the_rack`
- Preset sources: `manifests/episodes/rlinf_pi05_libero_pro_goal_swap_related_transfer.json` (`afc3125c3cb074b63d57d2f14ab5758011a2f3ca9c7c2ed984f8104e002aa9db`)
- Preset evolution launch: `launch/routes/libero_pro/rlinf_pi05_goal_swap.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero_pro/rlinf_pi05_goal_swap.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `libero_pro_goal_swap::open_the_middle_drawer_of_the_cabinet` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::open_the_middle_drawer_of_the_cabinet"}` |
| `libero_pro_goal_swap::open_the_top_drawer_and_put_the_bowl_inside` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::open_the_top_drawer_and_put_the_bowl_inside"}` |
| `libero_pro_goal_swap::push_the_plate_to_the_front_of_the_stove` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::push_the_plate_to_the_front_of_the_stove"}` |
| `libero_pro_goal_swap::put_the_bowl_on_the_plate` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::put_the_bowl_on_the_plate"}` |
| `libero_pro_goal_swap::put_the_bowl_on_the_stove` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::put_the_bowl_on_the_stove"}` |
| `libero_pro_goal_swap::put_the_bowl_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::put_the_bowl_on_top_of_the_cabinet"}` |
| `libero_pro_goal_swap::put_the_cream_cheese_in_the_bowl` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::put_the_cream_cheese_in_the_bowl"}` |
| `libero_pro_goal_swap::put_the_wine_bottle_on_the_rack` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::put_the_wine_bottle_on_the_rack"}` |
| `libero_pro_goal_swap::put_the_wine_bottle_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::put_the_wine_bottle_on_top_of_the_cabinet"}` |
| `libero_pro_goal_swap::turn_on_the_stove` | 10 | 300 | `{"task_id":"libero_pro_goal_swap::turn_on_the_stove"}` |

## `rlinf_pi05_libero_pro_goal_task`

- Route: RLinf pi0.5 + LIBERO-Pro goal task
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero_pro/rlinf_pi05_goal_task.sh`
- Profile: `configs/rlinf_pi05_libero_pro_goal_task.json` (`5f67e6cae2cabb7676ac4befd002e3c59d95a3273eac1e0094e86d8857ba1778`)
- Profile set: `libero_pro_goal_task`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT/tree/6222623f635769bfc73c9472e29fab9b7fd8e027) at `6222623f635769bfc73c9472e29fab9b7fd8e027`
- Full benchmark status: `ready`
- Metric: `equal_cell_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rlinf_pi05_libero_pro_goal_task_paper_v3_10_seed_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task LIBERO-Pro cell, not the eight-cell headline. It uses the released RLinf policy without the unreleased Harness memory agent.
- Route benchmark plan: `routes/libero_pro/rlinf_pi05_libero_pro_goal_task/benchmark_plan.json` (`183899228b68a759bc4520e6c6a76c5b7d74537f532df993fbb76abbb6c9357c`)
- Exact standard source: `manifests/benchmarks/rlinf_pi05_libero_pro_harness_paper_v3.json` (`03e6adde51c602740f3bf5c9f3d0e55640458ffe5d88d472c70ba730f78f0412`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `libero_pro_goal_task::open_the_middle_drawer_of_the_cabinet`, `libero_pro_goal_task::push_the_plate_to_the_front_of_the_stove`, `libero_pro_goal_task::put_the_cream_cheese_in_the_bowl`, `libero_pro_goal_task::put_the_wine_bottle_on_top_of_the_cabinet`, `libero_pro_goal_task::turn_on_the_stove`
- Preset held-out tasks: `libero_pro_goal_task::open_the_top_drawer_and_put_the_bowl_inside`, `libero_pro_goal_task::put_the_bowl_on_the_plate`, `libero_pro_goal_task::put_the_bowl_on_the_stove`, `libero_pro_goal_task::put_the_bowl_on_top_of_the_cabinet`, `libero_pro_goal_task::put_the_wine_bottle_on_the_rack`
- Preset sources: `manifests/episodes/rlinf_pi05_libero_pro_goal_task_related_transfer.json` (`97ac796f83b8aa5a8a59e5190a36d879668cdaeee906f5bc037624502491ac7d`)
- Preset evolution launch: `launch/routes/libero_pro/rlinf_pi05_goal_task.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero_pro/rlinf_pi05_goal_task.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `libero_pro_goal_task::open_the_middle_drawer_of_the_cabinet` | 10 | 300 | `{"task_id":"libero_pro_goal_task::open_the_middle_drawer_of_the_cabinet"}` |
| `libero_pro_goal_task::open_the_top_drawer_and_put_the_bowl_inside` | 10 | 300 | `{"task_id":"libero_pro_goal_task::open_the_top_drawer_and_put_the_bowl_inside"}` |
| `libero_pro_goal_task::push_the_plate_to_the_front_of_the_stove` | 10 | 300 | `{"task_id":"libero_pro_goal_task::push_the_plate_to_the_front_of_the_stove"}` |
| `libero_pro_goal_task::put_the_bowl_on_the_plate` | 10 | 300 | `{"task_id":"libero_pro_goal_task::put_the_bowl_on_the_plate"}` |
| `libero_pro_goal_task::put_the_bowl_on_the_stove` | 10 | 300 | `{"task_id":"libero_pro_goal_task::put_the_bowl_on_the_stove"}` |
| `libero_pro_goal_task::put_the_bowl_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"libero_pro_goal_task::put_the_bowl_on_top_of_the_cabinet"}` |
| `libero_pro_goal_task::put_the_cream_cheese_in_the_bowl` | 10 | 300 | `{"task_id":"libero_pro_goal_task::put_the_cream_cheese_in_the_bowl"}` |
| `libero_pro_goal_task::put_the_wine_bottle_on_the_rack` | 10 | 300 | `{"task_id":"libero_pro_goal_task::put_the_wine_bottle_on_the_rack"}` |
| `libero_pro_goal_task::put_the_wine_bottle_on_top_of_the_cabinet` | 10 | 300 | `{"task_id":"libero_pro_goal_task::put_the_wine_bottle_on_top_of_the_cabinet"}` |
| `libero_pro_goal_task::turn_on_the_stove` | 10 | 300 | `{"task_id":"libero_pro_goal_task::turn_on_the_stove"}` |

## `rlinf_pi05_libero_pro_object_swap`

- Route: RLinf pi0.5 + LIBERO-Pro object swap
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero_pro/rlinf_pi05_object_swap.sh`
- Profile: `configs/rlinf_pi05_libero_pro_object_swap.json` (`1d8932e5527afe39835f4671dba19790e1978d614e91a0f1dcc127c703fab509`)
- Profile set: `libero_pro_object_swap`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT/tree/6222623f635769bfc73c9472e29fab9b7fd8e027) at `6222623f635769bfc73c9472e29fab9b7fd8e027`
- Full benchmark status: `ready`
- Metric: `equal_cell_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rlinf_pi05_libero_pro_object_swap_paper_v3_10_seed_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task LIBERO-Pro cell, not the eight-cell headline. It uses the released RLinf policy without the unreleased Harness memory agent.
- Route benchmark plan: `routes/libero_pro/rlinf_pi05_libero_pro_object_swap/benchmark_plan.json` (`fbf28a1bda659d09bd74271871c1aaa2519f17200d12d7a83b6e5fcc7cfaaee6`)
- Exact standard source: `manifests/benchmarks/rlinf_pi05_libero_pro_harness_paper_v3.json` (`03e6adde51c602740f3bf5c9f3d0e55640458ffe5d88d472c70ba730f78f0412`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `libero_pro_object_swap::pick_up_the_alphabet_soup_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_bbq_sauce_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_cream_cheese_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_ketchup_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_salad_dressing_and_place_it_in_the_basket`
- Preset held-out tasks: `libero_pro_object_swap::pick_up_the_butter_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_milk_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_orange_juice_and_place_it_in_the_basket`, `libero_pro_object_swap::pick_up_the_tomato_sauce_and_place_it_in_the_basket`
- Preset sources: `manifests/episodes/rlinf_pi05_libero_pro_object_swap_related_transfer.json` (`d9f962e5eb781df2080617812bbf51d1792dc7e00952073a732ec24bd6addc1a`)
- Preset evolution launch: `launch/routes/libero_pro/rlinf_pi05_object_swap.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero_pro/rlinf_pi05_object_swap.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `libero_pro_object_swap::pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_alphabet_soup_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_bbq_sauce_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_butter_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_butter_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_chocolate_pudding_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_cream_cheese_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_cream_cheese_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_ketchup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_ketchup_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_milk_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_milk_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_orange_juice_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_orange_juice_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_salad_dressing_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_salad_dressing_and_place_it_in_the_basket"}` |
| `libero_pro_object_swap::pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_swap::pick_up_the_tomato_sauce_and_place_it_in_the_basket"}` |

## `rlinf_pi05_libero_pro_object_task`

- Route: RLinf pi0.5 + LIBERO-Pro object task
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero_pro/rlinf_pi05_object_task.sh`
- Profile: `configs/rlinf_pi05_libero_pro_object_task.json` (`d780a13069fa11a2f6f79b489e9ecba2ab5468923fddb7a51541e35e69d8bf58`)
- Profile set: `libero_pro_object_task`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT/tree/6222623f635769bfc73c9472e29fab9b7fd8e027) at `6222623f635769bfc73c9472e29fab9b7fd8e027`
- Full benchmark status: `ready`
- Metric: `equal_cell_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rlinf_pi05_libero_pro_object_task_paper_v3_10_seed_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task LIBERO-Pro cell, not the eight-cell headline. It uses the released RLinf policy without the unreleased Harness memory agent.
- Route benchmark plan: `routes/libero_pro/rlinf_pi05_libero_pro_object_task/benchmark_plan.json` (`f05e04359c9ee73d2a0f85d3a26177187d943886da3e9fcefaecfc66f5aedb6d`)
- Exact standard source: `manifests/benchmarks/rlinf_pi05_libero_pro_harness_paper_v3.json` (`03e6adde51c602740f3bf5c9f3d0e55640458ffe5d88d472c70ba730f78f0412`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `libero_pro_object_task::pick_up_the_alphabet_soup_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_bbq_sauce_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_cream_cheese_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_ketchup_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_salad_dressing_and_place_it_in_the_basket`
- Preset held-out tasks: `libero_pro_object_task::pick_up_the_butter_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_milk_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_orange_juice_and_place_it_in_the_basket`, `libero_pro_object_task::pick_up_the_tomato_sauce_and_place_it_in_the_basket`
- Preset sources: `manifests/episodes/rlinf_pi05_libero_pro_object_task_related_transfer.json` (`ca972e3d7e8bb9ec357fbfd35d36744e03fd62b1cd3d4e274913307ca5f52dea`)
- Preset evolution launch: `launch/routes/libero_pro/rlinf_pi05_object_task.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero_pro/rlinf_pi05_object_task.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `libero_pro_object_task::pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_alphabet_soup_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_bbq_sauce_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_butter_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_butter_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_chocolate_pudding_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_cream_cheese_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_cream_cheese_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_ketchup_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_ketchup_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_milk_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_milk_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_orange_juice_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_orange_juice_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_salad_dressing_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_salad_dressing_and_place_it_in_the_basket"}` |
| `libero_pro_object_task::pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 10 | 280 | `{"task_id":"libero_pro_object_task::pick_up_the_tomato_sauce_and_place_it_in_the_basket"}` |

## `rlinf_pi05_libero_pro_spatial_swap`

- Route: RLinf pi0.5 + LIBERO-Pro spatial swap
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero_pro/rlinf_pi05_spatial_swap.sh`
- Profile: `configs/rlinf_pi05_libero_pro_spatial_swap.json` (`1d9660955501a99a203e310f4a94605fe190794f301ad51b5178e4c2c85f333d`)
- Profile set: `libero_pro_spatial_swap`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT/tree/6222623f635769bfc73c9472e29fab9b7fd8e027) at `6222623f635769bfc73c9472e29fab9b7fd8e027`
- Full benchmark status: `ready`
- Metric: `equal_cell_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rlinf_pi05_libero_pro_spatial_swap_paper_v3_10_seed_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task LIBERO-Pro cell, not the eight-cell headline. It uses the released RLinf policy without the unreleased Harness memory agent.
- Route benchmark plan: `routes/libero_pro/rlinf_pi05_libero_pro_spatial_swap/benchmark_plan.json` (`acd5bc19066d2f79c162dcc6e8a12602a9772b9cd5b69db889bf560db3d28ec3`)
- Exact standard source: `manifests/benchmarks/rlinf_pi05_libero_pro_harness_paper_v3.json` (`03e6adde51c602740f3bf5c9f3d0e55640458ffe5d88d472c70ba730f78f0412`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `libero_pro_spatial_swap::pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`
- Preset held-out tasks: `libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`, `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`
- Preset sources: `manifests/episodes/rlinf_pi05_libero_pro_spatial_swap_related_transfer.json` (`407ec0df81a1886d00995b287d60fbc5b64ea259a9f1a498b7ccc339ffcf660a`)
- Preset evolution launch: `launch/routes/libero_pro/rlinf_pi05_spatial_swap.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero_pro/rlinf_pi05_spatial_swap.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `libero_pro_spatial_swap::pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_swap::pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}` |

## `rlinf_pi05_libero_pro_spatial_task`

- Route: RLinf pi0.5 + LIBERO-Pro spatial task
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero_pro/rlinf_pi05_spatial_task.sh`
- Profile: `configs/rlinf_pi05_libero_pro_spatial_task.json` (`1e19bcf9da9f3e82f4aebd6de7c86945b3cde514f8b881a2cd5ef1be5085e8d5`)
- Profile set: `libero_pro_spatial_task`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT/tree/6222623f635769bfc73c9472e29fab9b7fd8e027) at `6222623f635769bfc73c9472e29fab9b7fd8e027`
- Full benchmark status: `ready`
- Metric: `equal_cell_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rlinf_pi05_libero_pro_spatial_task_paper_v3_10_seed_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task LIBERO-Pro cell, not the eight-cell headline. It uses the released RLinf policy without the unreleased Harness memory agent.
- Route benchmark plan: `routes/libero_pro/rlinf_pi05_libero_pro_spatial_task/benchmark_plan.json` (`0a7a1b313c6bb6257529241bfb1082722fe043a9122a7ea0e28f00b1dd9a99c9`)
- Exact standard source: `manifests/benchmarks/rlinf_pi05_libero_pro_harness_paper_v3.json` (`03e6adde51c602740f3bf5c9f3d0e55640458ffe5d88d472c70ba730f78f0412`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `libero_pro_spatial_task::pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`
- Preset held-out tasks: `libero_pro_spatial_task::pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`, `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`
- Preset sources: `manifests/episodes/rlinf_pi05_libero_pro_spatial_task_related_transfer.json` (`a8cd409dd8868fb5208df82acff5e4991ab7d2f519a21157fb7b7c8b4038fd41`)
- Preset evolution launch: `launch/routes/libero_pro/rlinf_pi05_spatial_task.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero_pro/rlinf_pi05_spatial_task.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `libero_pro_spatial_task::pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"}` |
| `libero_pro_spatial_task::pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"libero_pro_spatial_task::pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}` |

## `rlinf_pi05_libero_spatial`

- Route: RLinf pi0.5 + LIBERO Spatial
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/rlinf_pi05_spatial.sh`
- Profile: `configs/rlinf_pi05_libero_spatial.json` (`9b1fc038007ab72940408dfc3fd51ba43a5b48d46e7f7e6848cf2ccde4c1d602`)
- Profile set: `libero_spatial`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT/tree/6222623f635769bfc73c9472e29fab9b7fd8e027) at `6222623f635769bfc73c9472e29fab9b7fd8e027`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `rlinf_pi05_libero_spatial_canonical_10_per_task_v1`
- Standard route rows: 100
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/rlinf_pi05_libero_spatial/benchmark_plan.json` (`5f9a9e54d4680f78f9f819ed2fa33055ec568378704db692c6add3c9c6495516`)
- Exact standard source: `manifests/benchmarks/rlinf_pi05_libero_standard.json` (`ebf9966972d174408d6563e380b82d6d7c3b2438723d8f459a733c1c3cad3e55`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`
- Preset held-out tasks: `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`
- Preset sources: `manifests/episodes/rlinf_pi05_libero_spatial_related_transfer.json` (`db338b2b36072f80ec9d39cbef847a3d5d0aae6a4f3c3a4c7d92b97f2aeee50a`)
- Preset evolution launch: `launch/routes/libero/rlinf_pi05_spatial.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/rlinf_pi05_spatial.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 10 | 220 | `{"task_id":"pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}` |

## `smolvla_robocerebra`

- Route: SmolVLA public substitute + RoboCerebra public-60
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/robocerebra/smolvla_public60.sh`
- Profile: `configs/smolvla_robocerebra.json` (`2212c13b028418fe5ac44b5715d7009d900b7e420ce3e8b194682dcb36b05bed`)
- Profile set: `robocerebra_public60`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [lerobot/smolvla_robocerebra](https://huggingface.co/lerobot/smolvla_robocerebra/tree/7ff416240ff73bda10a2b5dbd4245f72eaa959d0) at `7ff416240ff73bda10a2b5dbd4245f72eaa959d0`
- Full benchmark status: `ready_noncomparable`
- Metric: `equal_condition_case_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `robocerebra_released_anchor_resume_smolvla_v1`
- Standard route rows: 600
- Comparability: This uses a public SmolVLA substitute and released-protocol deviations, not the paper OpenVLA-OFT checkpoint.
- Route benchmark plan: `manifests/benchmarks/smolvla_robocerebra_public60_10_per_case.json` (`142d12bb479018f7ec1874846fb39e15500913c45f31c1e2f02f960b1142cf0c`)
- Exact standard source: `manifests/benchmarks/smolvla_robocerebra_public60_10_per_case.json` (`142d12bb479018f7ec1874846fb39e15500913c45f31c1e2f02f960b1142cf0c`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `robocerebra_public60::Ideal::case1`, `robocerebra_public60::Ideal::case2`, `robocerebra_public60::Ideal::case3`, `robocerebra_public60::Ideal::case5`, `robocerebra_public60::Ideal::case7`
- Preset held-out tasks: `robocerebra_public60::Ideal::case10`, `robocerebra_public60::Ideal::case4`, `robocerebra_public60::Ideal::case6`, `robocerebra_public60::Ideal::case8`, `robocerebra_public60::Ideal::case9`
- Preset sources: `manifests/episodes/smolvla_robocerebra_related_transfer.json` (`07c650b546e9a6444af9cf8161312a1e1b6d80256fc6f7fb999312d93df24b2f`)
- Preset evolution launch: `launch/routes/robocerebra/smolvla_public60.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/robocerebra/smolvla_public60.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This RoboCerebra route exposes RGB only, with no metric depth or camera calibration, and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `robocerebra_public60::Ideal::case1` | 10 | 885 | `{"task_id":"robocerebra_public60::Ideal::case1"}` |
| `robocerebra_public60::Ideal::case10` | 10 | 1935 | `{"task_id":"robocerebra_public60::Ideal::case10"}` |
| `robocerebra_public60::Ideal::case2` | 10 | 585 | `{"task_id":"robocerebra_public60::Ideal::case2"}` |
| `robocerebra_public60::Ideal::case3` | 10 | 1485 | `{"task_id":"robocerebra_public60::Ideal::case3"}` |
| `robocerebra_public60::Ideal::case4` | 10 | 885 | `{"task_id":"robocerebra_public60::Ideal::case4"}` |
| `robocerebra_public60::Ideal::case5` | 10 | 885 | `{"task_id":"robocerebra_public60::Ideal::case5"}` |
| `robocerebra_public60::Ideal::case6` | 10 | 885 | `{"task_id":"robocerebra_public60::Ideal::case6"}` |
| `robocerebra_public60::Ideal::case7` | 10 | 1035 | `{"task_id":"robocerebra_public60::Ideal::case7"}` |
| `robocerebra_public60::Ideal::case8` | 10 | 1035 | `{"task_id":"robocerebra_public60::Ideal::case8"}` |
| `robocerebra_public60::Ideal::case9` | 10 | 1635 | `{"task_id":"robocerebra_public60::Ideal::case9"}` |
| `robocerebra_public60::Memory_Execution::case1` | 10 | 1185 | `{"task_id":"robocerebra_public60::Memory_Execution::case1"}` |
| `robocerebra_public60::Memory_Execution::case10` | 10 | 2085 | `{"task_id":"robocerebra_public60::Memory_Execution::case10"}` |
| `robocerebra_public60::Memory_Execution::case2` | 10 | 1185 | `{"task_id":"robocerebra_public60::Memory_Execution::case2"}` |
| `robocerebra_public60::Memory_Execution::case3` | 10 | 1485 | `{"task_id":"robocerebra_public60::Memory_Execution::case3"}` |
| `robocerebra_public60::Memory_Execution::case4` | 10 | 1185 | `{"task_id":"robocerebra_public60::Memory_Execution::case4"}` |
| `robocerebra_public60::Memory_Execution::case5` | 10 | 1035 | `{"task_id":"robocerebra_public60::Memory_Execution::case5"}` |
| `robocerebra_public60::Memory_Execution::case6` | 10 | 1785 | `{"task_id":"robocerebra_public60::Memory_Execution::case6"}` |
| `robocerebra_public60::Memory_Execution::case7` | 10 | 1785 | `{"task_id":"robocerebra_public60::Memory_Execution::case7"}` |
| `robocerebra_public60::Memory_Execution::case8` | 10 | 2085 | `{"task_id":"robocerebra_public60::Memory_Execution::case8"}` |
| `robocerebra_public60::Memory_Execution::case9` | 10 | 2085 | `{"task_id":"robocerebra_public60::Memory_Execution::case9"}` |
| `robocerebra_public60::Memory_Exploration::case1` | 10 | 1785 | `{"task_id":"robocerebra_public60::Memory_Exploration::case1"}` |
| `robocerebra_public60::Memory_Exploration::case10` | 10 | 1635 | `{"task_id":"robocerebra_public60::Memory_Exploration::case10"}` |
| `robocerebra_public60::Memory_Exploration::case2` | 10 | 1485 | `{"task_id":"robocerebra_public60::Memory_Exploration::case2"}` |
| `robocerebra_public60::Memory_Exploration::case3` | 10 | 1785 | `{"task_id":"robocerebra_public60::Memory_Exploration::case3"}` |
| `robocerebra_public60::Memory_Exploration::case4` | 10 | 1485 | `{"task_id":"robocerebra_public60::Memory_Exploration::case4"}` |
| `robocerebra_public60::Memory_Exploration::case5` | 10 | 2085 | `{"task_id":"robocerebra_public60::Memory_Exploration::case5"}` |
| `robocerebra_public60::Memory_Exploration::case6` | 10 | 1785 | `{"task_id":"robocerebra_public60::Memory_Exploration::case6"}` |
| `robocerebra_public60::Memory_Exploration::case7` | 10 | 1485 | `{"task_id":"robocerebra_public60::Memory_Exploration::case7"}` |
| `robocerebra_public60::Memory_Exploration::case8` | 10 | 2235 | `{"task_id":"robocerebra_public60::Memory_Exploration::case8"}` |
| `robocerebra_public60::Memory_Exploration::case9` | 10 | 1935 | `{"task_id":"robocerebra_public60::Memory_Exploration::case9"}` |
| `robocerebra_public60::Mix::case1` | 10 | 1785 | `{"task_id":"robocerebra_public60::Mix::case1"}` |
| `robocerebra_public60::Mix::case10` | 10 | 1935 | `{"task_id":"robocerebra_public60::Mix::case10"}` |
| `robocerebra_public60::Mix::case2` | 10 | 1485 | `{"task_id":"robocerebra_public60::Mix::case2"}` |
| `robocerebra_public60::Mix::case3` | 10 | 1785 | `{"task_id":"robocerebra_public60::Mix::case3"}` |
| `robocerebra_public60::Mix::case4` | 10 | 1785 | `{"task_id":"robocerebra_public60::Mix::case4"}` |
| `robocerebra_public60::Mix::case5` | 10 | 2085 | `{"task_id":"robocerebra_public60::Mix::case5"}` |
| `robocerebra_public60::Mix::case6` | 10 | 1485 | `{"task_id":"robocerebra_public60::Mix::case6"}` |
| `robocerebra_public60::Mix::case7` | 10 | 1185 | `{"task_id":"robocerebra_public60::Mix::case7"}` |
| `robocerebra_public60::Mix::case8` | 10 | 1185 | `{"task_id":"robocerebra_public60::Mix::case8"}` |
| `robocerebra_public60::Mix::case9` | 10 | 1485 | `{"task_id":"robocerebra_public60::Mix::case9"}` |
| `robocerebra_public60::Observation_Mismatching::case1` | 10 | 885 | `{"task_id":"robocerebra_public60::Observation_Mismatching::case1"}` |
| `robocerebra_public60::Observation_Mismatching::case10` | 10 | 1935 | `{"task_id":"robocerebra_public60::Observation_Mismatching::case10"}` |
| `robocerebra_public60::Observation_Mismatching::case2` | 10 | 585 | `{"task_id":"robocerebra_public60::Observation_Mismatching::case2"}` |
| `robocerebra_public60::Observation_Mismatching::case3` | 10 | 1485 | `{"task_id":"robocerebra_public60::Observation_Mismatching::case3"}` |
| `robocerebra_public60::Observation_Mismatching::case4` | 10 | 885 | `{"task_id":"robocerebra_public60::Observation_Mismatching::case4"}` |
| `robocerebra_public60::Observation_Mismatching::case5` | 10 | 885 | `{"task_id":"robocerebra_public60::Observation_Mismatching::case5"}` |
| `robocerebra_public60::Observation_Mismatching::case6` | 10 | 885 | `{"task_id":"robocerebra_public60::Observation_Mismatching::case6"}` |
| `robocerebra_public60::Observation_Mismatching::case7` | 10 | 1035 | `{"task_id":"robocerebra_public60::Observation_Mismatching::case7"}` |
| `robocerebra_public60::Observation_Mismatching::case8` | 10 | 1035 | `{"task_id":"robocerebra_public60::Observation_Mismatching::case8"}` |
| `robocerebra_public60::Observation_Mismatching::case9` | 10 | 1635 | `{"task_id":"robocerebra_public60::Observation_Mismatching::case9"}` |
| `robocerebra_public60::Random_Disturbance::case1` | 10 | 885 | `{"task_id":"robocerebra_public60::Random_Disturbance::case1"}` |
| `robocerebra_public60::Random_Disturbance::case10` | 10 | 1935 | `{"task_id":"robocerebra_public60::Random_Disturbance::case10"}` |
| `robocerebra_public60::Random_Disturbance::case2` | 10 | 585 | `{"task_id":"robocerebra_public60::Random_Disturbance::case2"}` |
| `robocerebra_public60::Random_Disturbance::case3` | 10 | 1485 | `{"task_id":"robocerebra_public60::Random_Disturbance::case3"}` |
| `robocerebra_public60::Random_Disturbance::case4` | 10 | 885 | `{"task_id":"robocerebra_public60::Random_Disturbance::case4"}` |
| `robocerebra_public60::Random_Disturbance::case5` | 10 | 885 | `{"task_id":"robocerebra_public60::Random_Disturbance::case5"}` |
| `robocerebra_public60::Random_Disturbance::case6` | 10 | 885 | `{"task_id":"robocerebra_public60::Random_Disturbance::case6"}` |
| `robocerebra_public60::Random_Disturbance::case7` | 10 | 1035 | `{"task_id":"robocerebra_public60::Random_Disturbance::case7"}` |
| `robocerebra_public60::Random_Disturbance::case8` | 10 | 1035 | `{"task_id":"robocerebra_public60::Random_Disturbance::case8"}` |
| `robocerebra_public60::Random_Disturbance::case9` | 10 | 1635 | `{"task_id":"robocerebra_public60::Random_Disturbance::case9"}` |

## `xvla_calvin`

- Route: X-VLA + CALVIN ABC to D
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/calvin/xvla_abc_to_d.sh`
- Profile: `configs/xvla_calvin.json` (`3dc30be03c1483a2c32a2504e4646dc0a2a9da924b7fe809d0cb74269b0c382a`)
- Profile set: `calvin_abc_d_prefix1`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [2toINF/X-VLA-Calvin-ABC_D](https://huggingface.co/2toINF/X-VLA-Calvin-ABC_D/tree/d76710ee314ee1fa8506f421664c989b40bae415) at `d76710ee314ee1fa8506f421664c989b40bae415`
- Full benchmark status: `blocked_evaluator`
- Metric: `mean_completed_subtasks_per_sequence`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: not available
- Standard route rows: 1000
- Comparability: No full X-VLA-specific CALVIN benchmark result can be reported until the missing evaluator is implemented.
- Task-list source: `manifests/calvin_official_sequences.json` (file SHA-256 `13d0d886958ab45938bb2a6c7988dea4fdba9b5a7ff162ef29bfd0c0c656be66`)
- Blocker: X-VLA-specific 1,000-sequence benchmark evaluator/launcher not yet implemented

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This CALVIN route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `close_drawer` | not available | not available | `null` |
| `lift_blue_block_drawer` | not available | not available | `null` |
| `lift_blue_block_slider` | not available | not available | `null` |
| `lift_blue_block_table` | not available | not available | `null` |
| `lift_pink_block_drawer` | not available | not available | `null` |
| `lift_pink_block_slider` | not available | not available | `null` |
| `lift_pink_block_table` | not available | not available | `null` |
| `lift_red_block_drawer` | not available | not available | `null` |
| `lift_red_block_slider` | not available | not available | `null` |
| `lift_red_block_table` | not available | not available | `null` |
| `move_slider_left` | not available | not available | `null` |
| `move_slider_right` | not available | not available | `null` |
| `open_drawer` | not available | not available | `null` |
| `place_in_drawer` | not available | not available | `null` |
| `place_in_slider` | not available | not available | `null` |
| `push_blue_block_left` | not available | not available | `null` |
| `push_blue_block_right` | not available | not available | `null` |
| `push_into_drawer` | not available | not available | `null` |
| `push_pink_block_left` | not available | not available | `null` |
| `push_pink_block_right` | not available | not available | `null` |
| `push_red_block_left` | not available | not available | `null` |
| `push_red_block_right` | not available | not available | `null` |
| `rotate_blue_block_left` | not available | not available | `null` |
| `rotate_blue_block_right` | not available | not available | `null` |
| `rotate_pink_block_left` | not available | not available | `null` |
| `rotate_pink_block_right` | not available | not available | `null` |
| `rotate_red_block_left` | not available | not available | `null` |
| `rotate_red_block_right` | not available | not available | `null` |
| `stack_block` | not available | not available | `null` |
| `turn_off_led` | not available | not available | `null` |
| `turn_off_lightbulb` | not available | not available | `null` |
| `turn_on_led` | not available | not available | `null` |
| `turn_on_lightbulb` | not available | not available | `null` |
| `unstack_block` | not available | not available | `null` |

## `xvla_libero_goal`

- Route: X-VLA + LIBERO Goal
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/xvla_goal.sh`
- Profile: `configs/xvla_libero_goal.json` (`11f36eb60183627150e33d826ec16cff3e765c7af13050db92e44bd25fed2a42`)
- Profile set: `libero_goal`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [2toINF/X-VLA-Libero](https://huggingface.co/2toINF/X-VLA-Libero/tree/129e71460678b7236cee6fc9707f09d9fa0c3590) at `129e71460678b7236cee6fc9707f09d9fa0c3590`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `xvla_libero_goal_canonical_50_per_task_v1`
- Standard route rows: 500
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/xvla_libero_goal/benchmark_plan.json` (`5a93893d90671c34ea44291b817e6166b0a4d9625431e5d19b2de45185118964`)
- Exact standard source: `manifests/benchmarks/xvla_libero_standard.json` (`c9f2aa2715e983c81e82cc9458ce494477caf025b55d6cefc25e4e3ba250a930`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `open_the_middle_drawer_of_the_cabinet`, `put_the_bowl_on_the_stove`, `put_the_bowl_on_top_of_the_cabinet`, `put_the_wine_bottle_on_top_of_the_cabinet`
- Preset held-out tasks: `open_the_top_drawer_and_put_the_bowl_inside`, `put_the_bowl_on_the_plate`, `put_the_wine_bottle_on_the_rack`
- Preset sources: `manifests/episodes/xvla_libero_goal_transfer.json` (`146fe60e2fceeb4545325f863233585edaadddc9f572e33a85edf632b378fb2c`)
- Preset evolution launch: `launch/routes/libero/xvla_goal.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/xvla_goal.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `open_the_middle_drawer_of_the_cabinet` | 50 | 800 | `{"task_id":"open_the_middle_drawer_of_the_cabinet"}` |
| `open_the_top_drawer_and_put_the_bowl_inside` | 50 | 800 | `{"task_id":"open_the_top_drawer_and_put_the_bowl_inside"}` |
| `push_the_plate_to_the_front_of_the_stove` | 50 | 800 | `{"task_id":"push_the_plate_to_the_front_of_the_stove"}` |
| `put_the_bowl_on_the_plate` | 50 | 800 | `{"task_id":"put_the_bowl_on_the_plate"}` |
| `put_the_bowl_on_the_stove` | 50 | 800 | `{"task_id":"put_the_bowl_on_the_stove"}` |
| `put_the_bowl_on_top_of_the_cabinet` | 50 | 800 | `{"task_id":"put_the_bowl_on_top_of_the_cabinet"}` |
| `put_the_cream_cheese_in_the_bowl` | 50 | 800 | `{"task_id":"put_the_cream_cheese_in_the_bowl"}` |
| `put_the_wine_bottle_on_the_rack` | 50 | 800 | `{"task_id":"put_the_wine_bottle_on_the_rack"}` |
| `put_the_wine_bottle_on_top_of_the_cabinet` | 50 | 800 | `{"task_id":"put_the_wine_bottle_on_top_of_the_cabinet"}` |
| `turn_on_the_stove` | 50 | 800 | `{"task_id":"turn_on_the_stove"}` |

## `xvla_libero_long`

- Route: X-VLA + LIBERO Long
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/xvla_long.sh`
- Profile: `configs/xvla_libero_long.json` (`a5235353ae1882d075d993809ca44b9f433ac649f98bd8c94751a1f9f65ac5a6`)
- Profile set: `libero_10`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [2toINF/X-VLA-Libero](https://huggingface.co/2toINF/X-VLA-Libero/tree/129e71460678b7236cee6fc9707f09d9fa0c3590) at `129e71460678b7236cee6fc9707f09d9fa0c3590`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `xvla_libero_10_canonical_50_per_task_v1`
- Standard route rows: 500
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/xvla_libero_long/benchmark_plan.json` (`5c3bd0b1310592a62f4b10fd4324e6a614b39392d1ad03ac6c750cb94fd178ba`)
- Exact standard source: `manifests/benchmarks/xvla_libero_standard.json` (`c9f2aa2715e983c81e82cc9458ce494477caf025b55d6cefc25e4e3ba250a930`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`, `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`, `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`, `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
- Preset held-out tasks: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`, `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`
- Preset sources: `manifests/episodes/xvla_libero_long_transfer.json` (`61a16159a8281facb36c9b78cf7f098fdc4ae98c6871ff2747c96a6efb8c294c`)
- Preset evolution launch: `launch/routes/libero/xvla_long.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/xvla_long.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 50 | 900 | `{"task_id":"KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}` |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 50 | 900 | `{"task_id":"KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}` |
| `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 50 | 900 | `{"task_id":"KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"}` |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 50 | 900 | `{"task_id":"KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}` |
| `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 50 | 900 | `{"task_id":"LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 50 | 900 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"}` |
| `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 50 | 900 | `{"task_id":"LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"}` |
| `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 50 | 900 | `{"task_id":"LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"}` |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 50 | 900 | `{"task_id":"LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"}` |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 50 | 900 | `{"task_id":"STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"}` |

## `xvla_libero_object`

- Route: X-VLA + LIBERO Object
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/xvla_object.sh`
- Profile: `configs/xvla_libero_object.json` (`314385816d5d58f60892cfc63a8646ed197a7b883266388e44cf94ce96678f07`)
- Profile set: `libero_object`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [2toINF/X-VLA-Libero](https://huggingface.co/2toINF/X-VLA-Libero/tree/129e71460678b7236cee6fc9707f09d9fa0c3590) at `129e71460678b7236cee6fc9707f09d9fa0c3590`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `xvla_libero_object_canonical_50_per_task_v1`
- Standard route rows: 500
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/xvla_libero_object/benchmark_plan.json` (`62461966e4cd76ffa884d333f15f92b74d9fbca71f5b627b48968e17b65a0837`)
- Exact standard source: `manifests/benchmarks/xvla_libero_standard.json` (`c9f2aa2715e983c81e82cc9458ce494477caf025b55d6cefc25e4e3ba250a930`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`, `pick_up_the_bbq_sauce_and_place_it_in_the_basket`, `pick_up_the_cream_cheese_and_place_it_in_the_basket`, `pick_up_the_ketchup_and_place_it_in_the_basket`, `pick_up_the_salad_dressing_and_place_it_in_the_basket`
- Preset held-out tasks: `pick_up_the_butter_and_place_it_in_the_basket`, `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, `pick_up_the_milk_and_place_it_in_the_basket`, `pick_up_the_orange_juice_and_place_it_in_the_basket`, `pick_up_the_tomato_sauce_and_place_it_in_the_basket`
- Preset sources: `manifests/episodes/xvla_libero_object_transfer.json` (`9aa0aa049197200cad8d0c3bae1fca538a415eca7d97569c47d4adabf3d056e0`)
- Preset evolution launch: `launch/routes/libero/xvla_object.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/xvla_object.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_alphabet_soup_and_place_it_in_the_basket"}` |
| `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_bbq_sauce_and_place_it_in_the_basket"}` |
| `pick_up_the_butter_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_butter_and_place_it_in_the_basket"}` |
| `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_chocolate_pudding_and_place_it_in_the_basket"}` |
| `pick_up_the_cream_cheese_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_cream_cheese_and_place_it_in_the_basket"}` |
| `pick_up_the_ketchup_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_ketchup_and_place_it_in_the_basket"}` |
| `pick_up_the_milk_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_milk_and_place_it_in_the_basket"}` |
| `pick_up_the_orange_juice_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_orange_juice_and_place_it_in_the_basket"}` |
| `pick_up_the_salad_dressing_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_salad_dressing_and_place_it_in_the_basket"}` |
| `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 50 | 800 | `{"task_id":"pick_up_the_tomato_sauce_and_place_it_in_the_basket"}` |

## `xvla_libero_spatial`

- Route: X-VLA + LIBERO Spatial
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/libero/xvla_spatial.sh`
- Profile: `configs/xvla_libero.json` (`f28799a2524dc56316b5373cf9d43a573a56d13ea69b2d80d636b5b2452ad7c4`)
- Profile set: `libero_spatial`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [2toINF/X-VLA-Libero](https://huggingface.co/2toINF/X-VLA-Libero/tree/129e71460678b7236cee6fc9707f09d9fa0c3590) at `129e71460678b7236cee6fc9707f09d9fa0c3590`
- Full benchmark status: `ready`
- Metric: `equal_suite_task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `xvla_libero_spatial_canonical_50_per_task_v1`
- Standard route rows: 500
- Comparability: This launcher reports one standard 10-task suite, not the four-suite headline. The evolved agent uses additional frozen tools and must not be labeled as the raw policy.
- Route benchmark plan: `routes/libero/xvla_libero_spatial/benchmark_plan.json` (`f1dbb5c4e37f532c220be7c3e16a07fc25ffc8a24d49ffd8d832c5d03f5e1e5d`)
- Exact standard source: `manifests/benchmarks/xvla_libero_standard.json` (`c9f2aa2715e983c81e82cc9458ce494477caf025b55d6cefc25e4e3ba250a930`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`
- Preset held-out tasks: `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`, `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`
- Preset sources: `manifests/episodes/libero_spatial_transfer.json` (`5fb3ab13c040b78afefbd251e882e90d6b3bc6e013759a0c88ed7ecfc00da26c`)
- Preset evolution launch: `launch/routes/libero/xvla_spatial.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/libero/xvla_spatial.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This LIBERO route exposes no metric depth or camera calibration and has no Franka inverse-kinematics and trajectory executor for GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"}` |
| `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 50 | 800 | `{"task_id":"pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}` |

## `xvla_robotwin2`

- Route: X-VLA + RoboTwin 2.0 demo_clean
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/robotwin2/xvla_demo_clean.sh`
- Profile: `configs/xvla_robotwin2.json` (`dba77497508a188f9240cda24cb82f010d24b884d5c0adb18db85ff29d2053b5`)
- Profile set: `robotwin2_demo_clean`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [2toINF/X-VLA-RoboTwin2](https://huggingface.co/2toINF/X-VLA-RoboTwin2/tree/a157c580cfe6f9f445614490f3bec1b2f9ef9f18) at `a157c580cfe6f9f445614490f3bec1b2f9ef9f18`
- Full benchmark status: `requires_preparation`
- Metric: `equal_task_macro_success`
- Default resources: 2 GPUs, 2 workers per GPU, 4 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: not available
- Standard route rows: 5000
- Comparability: The prepared 50-task by 100-episode X-VLA protocol is not Harness RoboTwin C2R.
- Task-list source: `manifests/robotwin2.json` (file SHA-256 `18cc92b17d97237b6fcfc21abe8030771469d9031e60514eb1b8efb0c974e0c6`)
- Blocker: The expert-success-filtered 5,000-row benchmark plan must be prepared before launch

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | The official X-VLA route exposes no metric depth or calibrated grasp-to-dual-arm execution contract. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `adjust_bottle` | 100 | not available | `null` |
| `beat_block_hammer` | 100 | not available | `null` |
| `blocks_ranking_rgb` | 100 | not available | `null` |
| `blocks_ranking_size` | 100 | not available | `null` |
| `click_alarmclock` | 100 | not available | `null` |
| `click_bell` | 100 | not available | `null` |
| `dump_bin_bigbin` | 100 | not available | `null` |
| `grab_roller` | 100 | not available | `null` |
| `handover_block` | 100 | not available | `null` |
| `handover_mic` | 100 | not available | `null` |
| `hanging_mug` | 100 | not available | `null` |
| `lift_pot` | 100 | not available | `null` |
| `move_can_pot` | 100 | not available | `null` |
| `move_pillbottle_pad` | 100 | not available | `null` |
| `move_playingcard_away` | 100 | not available | `null` |
| `move_stapler_pad` | 100 | not available | `null` |
| `open_laptop` | 100 | not available | `null` |
| `open_microwave` | 100 | not available | `null` |
| `pick_diverse_bottles` | 100 | not available | `null` |
| `pick_dual_bottles` | 100 | not available | `null` |
| `place_a2b_left` | 100 | not available | `null` |
| `place_a2b_right` | 100 | not available | `null` |
| `place_bread_basket` | 100 | not available | `null` |
| `place_bread_skillet` | 100 | not available | `null` |
| `place_burger_fries` | 100 | not available | `null` |
| `place_can_basket` | 100 | not available | `null` |
| `place_cans_plasticbox` | 100 | not available | `null` |
| `place_container_plate` | 100 | not available | `null` |
| `place_dual_shoes` | 100 | not available | `null` |
| `place_empty_cup` | 100 | not available | `null` |
| `place_fan` | 100 | not available | `null` |
| `place_mouse_pad` | 100 | not available | `null` |
| `place_object_basket` | 100 | not available | `null` |
| `place_object_scale` | 100 | not available | `null` |
| `place_object_stand` | 100 | not available | `null` |
| `place_phone_stand` | 100 | not available | `null` |
| `place_shoe` | 100 | not available | `null` |
| `press_stapler` | 100 | not available | `null` |
| `put_bottles_dustbin` | 100 | not available | `null` |
| `put_object_cabinet` | 100 | not available | `null` |
| `rotate_qrcode` | 100 | not available | `null` |
| `scan_object` | 100 | not available | `null` |
| `shake_bottle` | 100 | not available | `null` |
| `shake_bottle_horizontally` | 100 | not available | `null` |
| `stack_blocks_three` | 100 | not available | `null` |
| `stack_blocks_two` | 100 | not available | `null` |
| `stack_bowls_three` | 100 | not available | `null` |
| `stack_bowls_two` | 100 | not available | `null` |
| `stamp_seal` | 100 | not available | `null` |
| `turn_switch` | 100 | not available | `null` |

## `xvla_simpler_google_va`

- Route: X-VLA + SimplerEnv Google Variant Aggregation
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/simpler/xvla_google_va.sh`
- Profile: `configs/xvla_simpler_google_va.json` (`6e33462f3f4954d61078afdda13ad8b31d49f41c2fc656792039083954fb2ba6`)
- Profile set: `simpler_google_va`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [2toINF/X-VLA-Google-Robot](https://huggingface.co/2toINF/X-VLA-Google-Robot/tree/afaad7ac52e483629e688f0c9c681cc58472d130) at `afaad7ac52e483629e688f0c9c681cc58472d130`
- Full benchmark status: `ready`
- Metric: `task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `xvla_google_va_extended_v1`
- Standard route rows: 1992
- Comparability: This is the complete five-task extended grid, not the four-task X-VLA paper-headline subset.
- Route benchmark plan: `manifests/benchmarks/xvla_simpler_google_va_extended_v1.json` (`f36a9b3c94eb77a8f9c52d188f944401b217f33a61a8855407a9b016392c09c5`)
- Exact standard source: `manifests/benchmarks/xvla_simpler_google_va_extended_v1.json` (`f36a9b3c94eb77a8f9c52d188f944401b217f33a61a8855407a9b016392c09c5`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `google_robot_open_drawer`
- Preset held-out tasks: `google_robot_close_drawer`
- Preset sources: `manifests/episodes/simpler_google_va_drawer_transfer_v2.json` (`e8f0a4aa5d869f4c8a003e2c3035f907ade623c63f7afdd5a00901f91e3cd929`)
- Preset evolution launch: `launch/routes/simpler/xvla_google_va.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/simpler/xvla_google_va.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This SimplerEnv route has no calibrated metric-depth observation and no controller that executes GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) | 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `google_robot_close_drawer` | 189 | 226 | `{"task_id":"google_robot_close_drawer"}` |
| `google_robot_move_near` | 600 | 160 | `{"task_id":"google_robot_move_near"}` |
| `google_robot_open_drawer` | 189 | 226 | `{"task_id":"google_robot_open_drawer"}` |
| `google_robot_pick_coke_can` | 825 | 160 | `{"task_id":"google_robot_pick_coke_can"}` |
| `google_robot_place_apple_in_closed_top_drawer` | 189 | 400 | `{"task_id":"google_robot_place_apple_in_closed_top_drawer"}` |

## `xvla_simpler_google_vm`

- Route: X-VLA + SimplerEnv Google Visual Matching
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/simpler/xvla_google_vm.sh`
- Profile: `configs/xvla_simpler_google_vm.json` (`e2e3d614da10d8db1aa0ca1dc23d62c6ad5e13a701920f277fd3e0bb8945c5a6`)
- Profile set: `simpler_google_vm`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [2toINF/X-VLA-Google-Robot](https://huggingface.co/2toINF/X-VLA-Google-Robot/tree/afaad7ac52e483629e688f0c9c681cc58472d130) at `afaad7ac52e483629e688f0c9c681cc58472d130`
- Full benchmark status: `ready`
- Metric: `task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `xvla_google_vm_extended_v1`
- Standard route rows: 864
- Comparability: This is the complete five-task extended grid, not the four-task X-VLA paper-headline subset.
- Route benchmark plan: `manifests/benchmarks/xvla_simpler_google_vm_extended_v1.json` (`9afb990bc340977f5985b195e7dda29efef61b659f5a2632b3c2883ec591e566`)
- Exact standard source: `manifests/benchmarks/xvla_simpler_google_vm_extended_v1.json` (`9afb990bc340977f5985b195e7dda29efef61b659f5a2632b3c2883ec591e566`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `google_robot_open_drawer`
- Preset held-out tasks: `google_robot_close_drawer`
- Preset sources: `manifests/episodes/simpler_google_vm_drawer_transfer_v1.json` (`56bb7d27e5876f4b7f3d21e7e62220a2bd8d60154e6405de1fbc1b77df0f6eab`)
- Preset evolution launch: `launch/routes/simpler/xvla_google_vm.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/simpler/xvla_google_vm.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This SimplerEnv route has no calibrated metric-depth observation and no controller that executes GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) | 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `google_robot_close_drawer` | 108 | 226 | `{"task_id":"google_robot_close_drawer"}` |
| `google_robot_move_near` | 240 | 160 | `{"task_id":"google_robot_move_near"}` |
| `google_robot_open_drawer` | 108 | 226 | `{"task_id":"google_robot_open_drawer"}` |
| `google_robot_pick_coke_can` | 300 | 160 | `{"task_id":"google_robot_pick_coke_can"}` |
| `google_robot_place_apple_in_closed_top_drawer` | 108 | 400 | `{"task_id":"google_robot_place_apple_in_closed_top_drawer"}` |

## `xvla_simpler_widowx`

- Route: X-VLA + SimplerEnv WidowX Visual Matching
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/simpler/xvla_widowx_vm.sh`
- Profile: `configs/xvla_simpler_widowx.json` (`f5e1b1cc5e69fa266d6a21e52e7ba8d6bf03b715e89833813afaa05d204e162a`)
- Profile set: `simpler_widowx_vm`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [2toINF/X-VLA-WidowX](https://huggingface.co/2toINF/X-VLA-WidowX/tree/8d7ea1aaa948665d44129a3ff488629b955fc0f9) at `8d7ea1aaa948665d44129a3ff488629b955fc0f9`
- Full benchmark status: `ready`
- Metric: `task_macro_success`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `xvla_widowx_vm_standard_v1`
- Standard route rows: 96
- Comparability: The route uses the disclosed absolute robot-base controller correction, not the untouched released default.
- Route benchmark plan: `manifests/benchmarks/xvla_simpler_widowx_vm_standard_v1.json` (`d3853e1bf6b64f5f2494352d90e534c955e5ac20ae9aecd728f8049fe9b87c37`)
- Exact standard source: `manifests/benchmarks/xvla_simpler_widowx_vm_standard_v1.json` (`d3853e1bf6b64f5f2494352d90e534c955e5ac20ae9aecd728f8049fe9b87c37`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `widowx_spoon_on_towel`, `widowx_stack_cube`
- Preset held-out tasks: `widowx_carrot_on_plate`, `widowx_put_eggplant_in_basket`
- Preset sources: `manifests/episodes/simpler_widowx_vm_transfer_v2.json` (`b9aeb58b62ee4365016cb889a54940e5cbea006922790d9a26353e13a59b011b`)
- Preset evolution launch: `launch/routes/simpler/xvla_widowx_vm.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/simpler/xvla_widowx_vm.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | This SimplerEnv route has no calibrated metric-depth observation and no controller that executes GraspGen poses. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) | 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `widowx_carrot_on_plate` | 24 | 1200 | `{"task_id":"widowx_carrot_on_plate"}` |
| `widowx_put_eggplant_in_basket` | 24 | 1200 | `{"task_id":"widowx_put_eggplant_in_basket"}` |
| `widowx_spoon_on_towel` | 24 | 1200 | `{"task_id":"widowx_spoon_on_towel"}` |
| `widowx_stack_cube` | 24 | 1200 | `{"task_id":"widowx_stack_cube"}` |

## `xvla_vlabench`

- Route: X-VLA + VLABench tracks 1-4
- Study role: suite, cell, or standalone route
- Launcher: `launch/routes/vlabench/xvla_tracks_1_4.sh`
- Profile: `configs/xvla_vlabench.json` (`11e3136153dbec6745989aaae6657df11d71507bf83ef02157bc3f4a41ca17ab`)
- Profile set: `vlabench_xvla_tracks_1_4`
- Seed scaffold: `scaffolds/volo_harness_seed`
- Low-level policy: [2toINF/X-VLA-VLABench](https://huggingface.co/2toINF/X-VLA-VLABench/tree/0995f2f51c9f2e29d78f20080948d25ce7e28d88) at `0995f2f51c9f2e29d78f20080948d25ce7e28d88`
- Full benchmark status: `ready`
- Metric: `equal_track_task_macro_progress_score`
- Default resources: 2 GPUs, 4 workers per GPU, 8 total workers, 2 policy servers, and 5 shared tool servers
- Candidate budget: 30
- Protocols: `xvla_vlabench_official_four_track_10ep_v1`
- Standard route rows: 400
- Comparability: The 400 rows are 40 track-task groups by 10 configurations; only 12 semantic task names are unique across tracks.
- Route benchmark plan: `manifests/benchmarks/xvla_vlabench_official_four_track_10ep.json` (`1d2759fed8331744d594b3307a2b982097c69d39dbb2a3b7ba38a66150344ac6`)
- Exact standard source: `manifests/benchmarks/xvla_vlabench_official_four_track_10ep.json` (`1d2759fed8331744d594b3307a2b982097c69d39dbb2a3b7ba38a66150344ac6`)
- Recommended related-transfer preset: `related` (`audited_from_pinned_legacy_episode_plans`)
- Preset evolve tasks: `track_1::select_fruit`, `track_1::select_mahjong`, `track_1::select_poker`
- Preset held-out tasks: `track_1::select_drink`, `track_3::select_nth_largest_poker`, `track_3::select_unique_type_mahjong`
- Preset sources: `manifests/episodes/vlabench_related.json` (`2069442a6df7fc86b68a070c51d971a975e53eb16275de60664ee02e08fa077f`)
- Preset evolution launch: `launch/routes/vlabench/xvla_tracks_1_4.sh RUN_ID --task-preset related --target-candidates 30`
- After all candidates complete, preset freeze and transfer: `launch/routes/vlabench/xvla_tracks_1_4.sh RUN_ID --task-preset related --target-candidates 30 --finalize --run-transfer`
- Transfer claim: Within-environment related-task transfer only; arbitrary disjoint task selections do not support this claim.

Starting-agent tools:

| Capability | Enabled | Model | Revision | Disabled reason |
|---|---:|---|---|---|
| detection | yes | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb | — |
| grasp | no | not available | not available | The released X-VLA VLABench client exposes RGB and proprioception without a frozen GraspGen execution contract. |
| language | yes | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd | — |
| pointing | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |
| segmentation | yes | [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3/tree/5eac5d508135b2f19adc3ef095efb7d393236f75) | 5eac5d508135b2f19adc3ef095efb7d393236f75 | — |
| vision | yes | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B/tree/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b) | e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b | — |

Selectable standard task units:

| `--evolve-task` / `--transfer-task` value | Standard rows | Horizons | Row selector |
|---|---:|---|---|
| `track_1::add_condiment` | 10 | 200 | `{"scenario_prefix":"track_1_config_","task_id":"add_condiment"}` |
| `track_1::insert_flower` | 10 | 200 | `{"scenario_prefix":"track_1_config_","task_id":"insert_flower"}` |
| `track_1::select_book` | 10 | 200 | `{"scenario_prefix":"track_1_config_","task_id":"select_book"}` |
| `track_1::select_chemistry_tube` | 10 | 200 | `{"scenario_prefix":"track_1_config_","task_id":"select_chemistry_tube"}` |
| `track_1::select_drink` | 10 | 200 | `{"scenario_prefix":"track_1_config_","task_id":"select_drink"}` |
| `track_1::select_fruit` | 10 | 200 | `{"scenario_prefix":"track_1_config_","task_id":"select_fruit"}` |
| `track_1::select_mahjong` | 10 | 200 | `{"scenario_prefix":"track_1_config_","task_id":"select_mahjong"}` |
| `track_1::select_painting` | 10 | 200 | `{"scenario_prefix":"track_1_config_","task_id":"select_painting"}` |
| `track_1::select_poker` | 10 | 100 | `{"scenario_prefix":"track_1_config_","task_id":"select_poker"}` |
| `track_1::select_toy` | 10 | 200 | `{"scenario_prefix":"track_1_config_","task_id":"select_toy"}` |
| `track_2::add_condiment` | 10 | 200 | `{"scenario_prefix":"track_2_config_","task_id":"add_condiment"}` |
| `track_2::insert_flower` | 10 | 200 | `{"scenario_prefix":"track_2_config_","task_id":"insert_flower"}` |
| `track_2::select_book` | 10 | 200 | `{"scenario_prefix":"track_2_config_","task_id":"select_book"}` |
| `track_2::select_chemistry_tube` | 10 | 200 | `{"scenario_prefix":"track_2_config_","task_id":"select_chemistry_tube"}` |
| `track_2::select_drink` | 10 | 200 | `{"scenario_prefix":"track_2_config_","task_id":"select_drink"}` |
| `track_2::select_fruit` | 10 | 200 | `{"scenario_prefix":"track_2_config_","task_id":"select_fruit"}` |
| `track_2::select_mahjong` | 10 | 200 | `{"scenario_prefix":"track_2_config_","task_id":"select_mahjong"}` |
| `track_2::select_painting` | 10 | 200 | `{"scenario_prefix":"track_2_config_","task_id":"select_painting"}` |
| `track_2::select_poker` | 10 | 100 | `{"scenario_prefix":"track_2_config_","task_id":"select_poker"}` |
| `track_2::select_toy` | 10 | 200 | `{"scenario_prefix":"track_2_config_","task_id":"select_toy"}` |
| `track_3::add_condiment` | 10 | 200 | `{"scenario_prefix":"track_3_config_","task_id":"add_condiment"}` |
| `track_3::insert_flower` | 10 | 200 | `{"scenario_prefix":"track_3_config_","task_id":"insert_flower"}` |
| `track_3::select_book` | 10 | 200 | `{"scenario_prefix":"track_3_config_","task_id":"select_book"}` |
| `track_3::select_chemistry_tube` | 10 | 200 | `{"scenario_prefix":"track_3_config_","task_id":"select_chemistry_tube"}` |
| `track_3::select_drink` | 10 | 200 | `{"scenario_prefix":"track_3_config_","task_id":"select_drink"}` |
| `track_3::select_fruit` | 10 | 200 | `{"scenario_prefix":"track_3_config_","task_id":"select_fruit"}` |
| `track_3::select_nth_largest_poker` | 10 | 100 | `{"scenario_prefix":"track_3_config_","task_id":"select_nth_largest_poker"}` |
| `track_3::select_painting` | 10 | 200 | `{"scenario_prefix":"track_3_config_","task_id":"select_painting"}` |
| `track_3::select_toy` | 10 | 200 | `{"scenario_prefix":"track_3_config_","task_id":"select_toy"}` |
| `track_3::select_unique_type_mahjong` | 10 | 200 | `{"scenario_prefix":"track_3_config_","task_id":"select_unique_type_mahjong"}` |
| `track_4::add_condiment` | 10 | 200 | `{"scenario_prefix":"track_4_config_","task_id":"add_condiment"}` |
| `track_4::insert_flower` | 10 | 200 | `{"scenario_prefix":"track_4_config_","task_id":"insert_flower"}` |
| `track_4::select_book` | 10 | 200 | `{"scenario_prefix":"track_4_config_","task_id":"select_book"}` |
| `track_4::select_chemistry_tube` | 10 | 200 | `{"scenario_prefix":"track_4_config_","task_id":"select_chemistry_tube"}` |
| `track_4::select_drink` | 10 | 200 | `{"scenario_prefix":"track_4_config_","task_id":"select_drink"}` |
| `track_4::select_fruit` | 10 | 200 | `{"scenario_prefix":"track_4_config_","task_id":"select_fruit"}` |
| `track_4::select_mahjong` | 10 | 200 | `{"scenario_prefix":"track_4_config_","task_id":"select_mahjong"}` |
| `track_4::select_painting` | 10 | 200 | `{"scenario_prefix":"track_4_config_","task_id":"select_painting"}` |
| `track_4::select_poker` | 10 | 100 | `{"scenario_prefix":"track_4_config_","task_id":"select_poker"}` |
| `track_4::select_toy` | 10 | 200 | `{"scenario_prefix":"track_4_config_","task_id":"select_toy"}` |

## Blocked and not-ready routes

Blocked routes have no production wrapper.

| Route | Status | Launcher | Task source | Trials | Horizons | Metric | Protocol | Blocker |
|---|---|---|---|---|---|---|---|---|
| `cap_x_exact` | `external_baseline` | not available | not available | not available | not available | not available | not available | The exact local 39-task CaP-X benchmark/provider route is not integrated |
| `dreamzero_robolab120` | `component_only` | not available | not available | not available | not available | not available | not available | The public checkpoint is pinned, but no local backend or full RoboLab route is integrated |
| `harness_vla_exact` | `blocked` | not available | not available | not available | not available | not available | not available | No public Harness-VLA agent implementation, memory artifacts, or exact paper policy services |
| `lingbot_vla_v2_robotwin` | `blocked` | not available | not available | not available | not available | not available | not available | The public artifact is a base model; no pinned RoboTwin-post-trained checkpoint is released |
| `molmoact2_droid_robolab120` | `simulator_blocked` | not available | not available | not available | not available | not available | not available | RoboLab requires explicit NVIDIA Omniverse EULA acceptance and has no completed simulator episode |
| `molmobot_robolab120` | `simulator_blocked` | not available | not available | not available | not available | not available | not available | RoboLab requires explicit NVIDIA Omniverse EULA acceptance and has no completed simulator episode |
| `openpi_droid_velocity_robolab120` | `incompatible_action_space` | not available | not available | not available | not available | not available | not available | Generic OpenPI DROID velocity checkpoints do not match the absolute-joint RoboLab evaluator; only the explicit joint-position checkpoints are valid candidates |
| `openpi_pi05_droid_jointpos_robolab120` | `simulator_blocked` | not available | not available | not available | not available | not available | not available | The policy service is implemented, but RoboLab requires explicit NVIDIA Omniverse EULA acceptance and has no completed simulator episode |
| `openpi_pi0_fast_droid_jointpos_robolab120` | `simulator_blocked` | not available | not available | not available | not available | not available | not available | The policy service is implemented, but RoboLab requires explicit NVIDIA Omniverse EULA acceptance and has no completed simulator episode |
| `paper_openvla_oft_robocerebra` | `blocked` | not available | not available | not available | not available | not available | not available | The paper OpenVLA-OFT checkpoint and exact VoLo-specific RoboCerebra agent protocol are not public; SmolVLA is tracked separately as a substitute |
| `robovolo_benchmark` | `blocked` | not available | not available | not available | not available | not available | not available | The RoboVoLo benchmark package, exact task manifest, resets, and executable evaluator are not public |
| `tiptop_exact` | `external_baseline` | not available | not available | not available | not available | not available | not available | The exact local 28-task TiPToP benchmark/provider route and non-commercial dependencies are not integrated |
| `voloagent_exact` | `blocked` | not available | not available | not available | not available | not available | not available | No public VoLoAgent implementation, RoboVoLo task package, planner state, or complete policy/tool checkpoint set |
| `xvla_navsim` | `blocked` | not available | not available | not available | not available | not available | not available | The public X-VLA collection has no NAVSIM policy checkpoint |

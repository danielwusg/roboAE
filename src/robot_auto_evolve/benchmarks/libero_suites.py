from __future__ import annotations

from types import MappingProxyType


LIBERO_SUITE_TASKS = MappingProxyType(
    {
        "libero_spatial": (
            "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
            "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
            "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
            "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
            "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
            "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
            "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
            "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
            "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
            "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
        ),
        "libero_object": (
            "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
            "pick_up_the_cream_cheese_and_place_it_in_the_basket",
            "pick_up_the_salad_dressing_and_place_it_in_the_basket",
            "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
            "pick_up_the_ketchup_and_place_it_in_the_basket",
            "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
            "pick_up_the_butter_and_place_it_in_the_basket",
            "pick_up_the_milk_and_place_it_in_the_basket",
            "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
            "pick_up_the_orange_juice_and_place_it_in_the_basket",
        ),
        "libero_goal": (
            "open_the_middle_drawer_of_the_cabinet",
            "put_the_bowl_on_the_stove",
            "put_the_wine_bottle_on_top_of_the_cabinet",
            "open_the_top_drawer_and_put_the_bowl_inside",
            "put_the_bowl_on_top_of_the_cabinet",
            "push_the_plate_to_the_front_of_the_stove",
            "put_the_cream_cheese_in_the_bowl",
            "turn_on_the_stove",
            "put_the_bowl_on_the_plate",
            "put_the_wine_bottle_on_the_rack",
        ),
        "libero_10": (
            "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
            "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
            "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
            "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
            "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
            "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
            "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
            "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
            "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
            "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        ),
    }
)

LIBERO_TASK_SUITE = MappingProxyType(
    {
        task: suite
        for suite, tasks in LIBERO_SUITE_TASKS.items()
        for task in tasks
    }
)

XVLA_LIBERO_PROTOCOLS = MappingProxyType(
    {
        "libero_spatial": MappingProxyType(
            {
                "libero_spatial_meta_transfer_production_v2": 800,
                "libero_spatial_end_to_end_smoke_v3": 9,
                "xvla_libero_spatial_canonical_50_per_task_v1": 800,
            }
        ),
        "libero_object": MappingProxyType(
            {
                "libero_object_meta_transfer_production_v1": 800,
                "libero_object_end_to_end_smoke_v1": 9,
                "xvla_libero_object_canonical_50_per_task_v1": 800,
            }
        ),
        "libero_goal": MappingProxyType(
            {
                "libero_goal_meta_transfer_production_v1": 800,
                "libero_goal_end_to_end_smoke_v1": 9,
                "xvla_libero_goal_canonical_50_per_task_v1": 800,
            }
        ),
        "libero_10": MappingProxyType(
            {
                "libero_10_meta_transfer_production_v1": 900,
                "libero_10_end_to_end_smoke_v1": 9,
                "xvla_libero_10_canonical_50_per_task_v1": 900,
            }
        ),
    }
)

PI05_LIBERO_PROTOCOLS = MappingProxyType(
    {
        "libero_spatial": MappingProxyType(
            {
                "pi05_lerobot_libero_spatial_transfer_v1": 280,
                "pi05_lerobot_libero_spatial_smoke_v1": 9,
                "pi05_lerobot_libero_spatial_canonical_10_per_task_v1": 280,
            }
        ),
        "libero_object": MappingProxyType(
            {
                "pi05_lerobot_libero_object_transfer_v1": 280,
                "pi05_lerobot_libero_object_smoke_v1": 9,
                "pi05_lerobot_libero_object_canonical_10_per_task_v1": 280,
            }
        ),
        "libero_goal": MappingProxyType(
            {
                "pi05_lerobot_libero_goal_transfer_v1": 300,
                "pi05_lerobot_libero_goal_smoke_v1": 9,
                "pi05_lerobot_libero_goal_canonical_10_per_task_v1": 300,
            }
        ),
        "libero_10": MappingProxyType(
            {
                "pi05_lerobot_libero_10_transfer_v1": 520,
                "pi05_lerobot_libero_10_smoke_v1": 9,
                "pi05_lerobot_libero_10_canonical_10_per_task_v1": 520,
            }
        ),
    }
)

RLINF_PI05_LIBERO_PROTOCOLS = MappingProxyType(
    {
        "libero_spatial": MappingProxyType(
            {
                "rlinf_pi05_libero_spatial_related_transfer_v1": 220,
                "rlinf_pi05_libero_spatial_canonical_10_per_task_v1": 220,
                "rlinf_pi05_libero_spatial_canonical_one_step_smoke_v1": 1,
            }
        ),
        "libero_object": MappingProxyType(
            {
                "rlinf_pi05_libero_object_related_transfer_v1": 280,
                "rlinf_pi05_libero_object_canonical_10_per_task_v1": 280,
                "rlinf_pi05_libero_object_canonical_one_step_smoke_v1": 1,
            }
        ),
        "libero_goal": MappingProxyType(
            {
                "rlinf_pi05_libero_goal_related_transfer_v1": 300,
                "rlinf_pi05_libero_goal_canonical_10_per_task_v1": 300,
                "rlinf_pi05_libero_goal_canonical_one_step_smoke_v1": 1,
            }
        ),
        "libero_10": MappingProxyType(
            {
                "rlinf_pi05_libero_long_related_transfer_v1": 520,
                "rlinf_pi05_libero_long_canonical_10_per_task_v1": 520,
                "rlinf_pi05_libero_long_canonical_one_step_smoke_v1": 1,
            }
        ),
    }
)

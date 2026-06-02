#!/usr/bin/env python3
"""Compute route-weighted ClassIC-SR FLOPs using the repository convention."""
import argparse

BRANCH_FLOPS_M = {
    "classic_sr_easy": 7.9,
    "classic_sr_medium": 102.9,
    "classic_sr_hard": 468.0,
}
CLASSIFIER_FLOPS_M = 7.4


def route_weighted_flops(easy, medium, hard):
    total = easy + medium + hard
    if total <= 0:
        raise ValueError("route counts must sum to a positive value")
    return (
        BRANCH_FLOPS_M["classic_sr_easy"] * easy
        + BRANCH_FLOPS_M["classic_sr_medium"] * medium
        + BRANCH_FLOPS_M["classic_sr_hard"] * hard
    ) / total + CLASSIFIER_FLOPS_M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", nargs=3, type=int, metavar=("EASY", "MEDIUM", "HARD"), required=True)
    args = parser.parse_args()
    flops = route_weighted_flops(*args.route)
    print("route={}/{}/{} weighted_FLOPs={:.6f}M percent_vs_hard={:.6f}".format(*args.route, flops, flops / 468.0))


if __name__ == "__main__":
    main()

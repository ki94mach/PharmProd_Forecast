"""``python -m pkg.research`` → feature-experiment CLI help."""


def main() -> int:
    print(
        "Research feature experiments on frozen benchmark v1:\n"
        "  python -m pkg.research.evaluate_features\n"
        "\n"
        "Experiments: F0 (frozen), F1A (+demand), F1B (+human reliability), F1C (both)\n"
        "Audit:   python -m pkg.research.audit_f1\n"
        "F2:      python -m pkg.research.evaluate_f2\n"
        "Ablation: python -m pkg.research.evaluate_feature_ablation\n"
        "F3A:     python -m pkg.research.evaluate_f3a\n"
        "API: from pkg.research import compare_feature_experiments\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

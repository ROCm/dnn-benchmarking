# Contributing to dnn-benchmarking

Thanks for contributing to dnn-benchmarking.

## Reporting issues

Use [GitHub Issues](../../issues) for bug reports and feature requests. Include a clear description, reproduction steps, and the relevant OS, Python, ROCm, and PyTorch versions.

For security vulnerabilities, do not open a public issue. Follow [SECURITY.md](SECURITY.md).

## Development workflow

1. Fork the repository and create a branch from `main`.
2. Set up a development environment with `python3 setup_env.py --workspace .workspace`.
3. Make the change, add or update tests for observable behavior, and update documentation when behavior changes.
4. Run the relevant tests. At minimum, run `pytest -m "not gpu"` for Python changes that do not require hardware.
5. Open a pull request against `main`. Explain the change, its motivation, validation, and any related issue.
6. Ensure all required CI checks pass and request review from the applicable CODEOWNERS.

## License

By contributing, you agree that your contribution is licensed under the repository's [MIT License](LICENSE.md).

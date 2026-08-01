from torchdire.theory.verifier import verify_qgfd_theorems, QGFDTheoremVerifier


def test_theorems():
    verifier = QGFDTheoremVerifier()
    results = verifier.run_all(verbose=True)
    for name, passed in results.items():
        assert passed, f"Theoretical verification failed for {name}"


if __name__ == "__main__":
    test_theorems()
    print("All theoretical verification tests passed successfully!")

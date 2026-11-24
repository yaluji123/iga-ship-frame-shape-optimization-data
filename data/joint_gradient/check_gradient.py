"""Re-solve the moving skew joint and check the recorded adjoints and omissions."""

from pathlib import Path
import json

from moving_skew_gradient import np, run_study

HERE = Path(__file__).resolve().parent


def check():
    recorded = json.loads((HERE/"moving_skew_gradient.json").read_text(encoding="utf8"))
    computed = run_study()
    base = recorded["base"]
    current = computed["base"]

    for key in ("response", "adjoint_derivative", "eta_shape_direction", "chain_length_direction",
                "endpoint_S_direction_norm", "spectral_gaps"):
        np.testing.assert_allclose(current[key], base[key], rtol=3e-6, atol=2e-13, err_msg=key)

    assert base["free_dofs"]==current["free_dofs"]==162
    assert base["sample_count"]==current["sample_count"]==1600
    assert base["free_residual"]<1e-9 and base["transpose_adjoint_residual"]<1e-9
    assert current["skew_fraction"]>1e-4
    assert len(recorded["real_differences"])==len(computed["real_differences"])==4

    for ref, fresh in zip(recorded["real_differences"],computed["real_differences"],strict=True):
        h=ref["h"]
        assert h==fresh["h"]
        fd=(np.asarray(ref["plus"]["response"])-ref["minus"]["response"])/(2*h)
        err=abs(fd-base["adjoint_derivative"])
        taylor=abs(np.asarray(ref["plus"]["response"])-base["response"]-h*np.asarray(base["adjoint_derivative"]))
        np.testing.assert_allclose(fd,ref["centered_fd"],rtol=1e-13,atol=2e-15)
        np.testing.assert_allclose(err,ref["absolute_error"],rtol=1e-13,atol=2e-15)
        np.testing.assert_allclose(taylor,ref["forward_Taylor_remainder"],rtol=1e-13,atol=2e-15)
        np.testing.assert_allclose(fd,fresh["centered_fd"],rtol=3e-6,atol=2e-13)

    for k,ref in recorded["omissions"].items():
        new=computed["omissions"][k]
        omitted=np.asarray(ref["change_from_full_derivative"])
        np.testing.assert_allclose(new["change_from_full_derivative"],omitted,rtol=3e-6,atol=2e-14,err_msg=k)
        np.testing.assert_allclose(omitted+base["adjoint_derivative"],ref["omitted_derivative"],rtol=1e-13,atol=2e-15)
        assert ref["omitted_derivative_matrix_frobenius_norm"]>0
        assert min(ref["omission_difference_to_finest_fd_error"])>100

    print("Moving trim and receiving trace, spectral calibration and skew adjoints verified.")
    print("Four real perturbations reproduce the shape derivatives and Taylor behavior. The same-state derivative omissions are detectable.")


if __name__ == "__main__":
    check()

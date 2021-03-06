import re
import warnings

import multipledispatch
import pytest
import tensorflow as tf

import gpflow


class A1:
    pass


class A2(A1):
    pass


class B1:
    pass


class B2(B1):
    pass


def test_our_multipledispatch():
    test_fn = gpflow.utilities.Dispatcher("test_fn")

    @test_fn.register(A1, B1)
    def test_a1_b1(x, y):
        return "a1-b1"

    @test_fn.register(A2, B1)
    def test_a2_b1(x, y):
        return "a2-b1"

    @test_fn.register(A1, B2)
    def test_a1_b2(x, y):
        return "a1-b2"

    assert test_fn(A1(), B1()) == "a1-b1"
    assert test_fn(A2(), B1()) == "a2-b1"
    assert test_fn(A1(), B2()) == "a1-b2"

    # test the ambiguous case:

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        assert test_fn(A2(), B2()) == "a1-b2"  # the last definition wins

        assert len(w) == 1
        assert issubclass(w[0].category, multipledispatch.conflict.AmbiguityWarning)

    # test that adding the child-child definition removes ambiguity warning:

    @test_fn.register(A2, B2)
    def test_a2_b2(x, y):
        return "a2-b2"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        assert test_fn(A2(), B2()) == "a2-b2"

        assert len(w) == 0


@pytest.mark.parametrize(
    "Dispatcher, expect_autograph_warning",
    [(multipledispatch.Dispatcher, True), (gpflow.utilities.Dispatcher, False),],
)
def test_dispatcher_autograph_warnings(capsys, Dispatcher, expect_autograph_warning):
    tf.autograph.set_verbosity(
        0, alsologtostdout=True
    )  # to be able to capture it using capsys

    test_fn = Dispatcher("test_fn")

    # generator would only be invoked when defining for base class...
    @test_fn.register(gpflow.inducing_variables.InducingVariables)
    def test_iv(x):
        return tf.reduce_sum(x.Z)

    test_fn_jit = tf.function(test_fn)  # with autograph=True by default

    # ...but calling using subclass
    result = test_fn_jit(gpflow.inducing_variables.InducingPoints([1.0, 2.0]))
    assert result.numpy() == 3.0  # expect computation to work either way

    captured = capsys.readouterr()

    tf_warning = "WARNING:.*Entity .* appears to be a generator function. It will not be converted by AutoGraph."
    assert bool(re.match(tf_warning, captured.out)) == expect_autograph_warning

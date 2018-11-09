import pytest
import numpy as np

from emukit.core import DiscreteParameter, InformationSourceParameter
from emukit.core.discrete_parameter import create_discrete_parameter_for_categories


def test_discrete_parameter():
    param = DiscreteParameter('x', [0, 1, 2])
    assert param.check_in_domain(np.array([0, 1])) is True
    assert param.check_in_domain(np.array([3])) is False


def test_single_value_in_domain_discrete_parameter():
    param = DiscreteParameter('x', [0, 1, 2])
    assert param.check_in_domain(0) is True
    assert param.check_in_domain(3) is False


@pytest.fixture
def categories():
    return ['red', 'blue', 'green']

@pytest.fixture
def encodings():
    return [1, 2, 3]

def test_create_discrete_parameter_for_categories_errors(categories, encodings):
    encodings_too_long = encodings + [4]
    with pytest.raises(ValueError):
        create_discrete_parameter_for_categories('x', categories, encodings_too_long)

    encodings_too_short = encodings[:2]
    with pytest.raises(ValueError):
        create_discrete_parameter_for_categories('x', categories, encodings_too_short)

    encodings_with_duplicate = encodings
    encodings_with_duplicate[-1] = encodings_with_duplicate[0]
    with pytest.raises(ValueError):
        create_discrete_parameter_for_categories('x', categories, encodings_with_duplicate)

def test_create_discrete_parameter_for_categories_with_encodings(categories, encodings):
    param, forward_map, reverse_map = create_discrete_parameter_for_categories('x', categories, encodings)

    assert param.name == 'x'
    assert sorted(param.domain) == sorted(encodings)
    assert sorted(forward_map.keys()) == sorted(categories)
    assert sorted(forward_map.values()) == sorted(encodings)
    assert sorted(reverse_map.keys()) == sorted(encodings)
    assert sorted(reverse_map.values()) == sorted(categories)

def test_create_discrete_parameter_for_categories_without_encodings(categories):
    expected_encodings = [1, 10, 100]

    param, forward_map, reverse_map = create_discrete_parameter_for_categories('x', categories)

    assert param.name == 'x'
    assert sorted(param.domain) == sorted(expected_encodings)
    assert sorted(forward_map.keys()) == sorted(categories)
    assert sorted(forward_map.values()) == sorted(expected_encodings)
    assert sorted(reverse_map.keys()) == sorted(expected_encodings)
    assert sorted(reverse_map.values()) == sorted(categories)
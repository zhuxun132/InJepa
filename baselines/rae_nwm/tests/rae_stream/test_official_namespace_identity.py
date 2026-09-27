"""Official namespace packages need rooted search-location identity, not fake init files."""
import importlib.machinery
import sys
from types import ModuleType
import pytest
from rae_stream.official_factory import _import_planning_module


def install(monkeypatch,root,locations):
    planning=ModuleType('planning_eval')
    planning.__spec__=importlib.machinery.ModuleSpec('planning_eval',None,origin=str(root/'planning_eval.py'))
    monkeypatch.setitem(sys.modules,'planning_eval',planning)
    namespace=ModuleType('RAE')
    namespace.__spec__=importlib.machinery.ModuleSpec('RAE',None,is_package=True)
    namespace.__spec__.submodule_search_locations=[str(path) for path in locations]
    monkeypatch.setitem(sys.modules,'RAE',namespace)
    return planning


def test_official_namespace_locations_inside_exact_package_directory_accepted(tmp_path,monkeypatch):
    package=tmp_path/'RAE';package.mkdir()
    planning=install(monkeypatch,tmp_path,[package])
    assert _import_planning_module(tmp_path) is planning
    assert not (package/'__init__.py').exists()

@pytest.mark.parametrize('kind',['outside','wrong_name','empty','mixed'])
def test_namespace_unknown_or_foreign_locations_fail_closed(tmp_path,monkeypatch,kind):
    good=tmp_path/'RAE';good.mkdir()
    wrong=tmp_path/'unrelated';wrong.mkdir()
    outside=tmp_path.parent/'foreign_namespace';outside.mkdir(exist_ok=True)
    locations={'outside':[outside],'wrong_name':[wrong],'empty':[],'mixed':[good,outside]}[kind]
    install(monkeypatch,tmp_path,locations)
    with pytest.raises(RuntimeError):_import_planning_module(tmp_path)

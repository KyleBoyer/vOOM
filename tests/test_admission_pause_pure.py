from types import SimpleNamespace as NS

import pytest

from runtime.admission_pause import AdmissionPause


def run(pause, *, recovery=100, pressure=False):
    now=[0.0]; clears=[]
    def sample():
        return (10, 50 if pressure else 100, 110 if now[0] >= recovery else 90, 100)
    result=pause.wait(sample,critical=60,reason='test',clear_cache=lambda: clears.append(1),
        clock=lambda: now[0],sleep=lambda n: now.__setitem__(0,now[0]+n),
        swap=lambda: NS(used=0,sout=0))
    return result,now[0],clears


def test_default_off_has_no_samples_waits_or_cache_operations():
    assert run(AdmissionPause()) == (None,0,[])


def test_recovers_only_when_original_predicate_is_safe():
    pause=AdmissionPause(True)
    result,elapsed,clears=run(pause,recovery=2)
    assert result[3] <= result[2] and elapsed==2 and len(clears)==8
    assert pause.remaining_s==28


def test_per_call_and_lifetime_limits_are_not_reset():
    pause=AdmissionPause(True)
    for _ in range(6):
        result,elapsed,_=run(pause)
        assert result is None and elapsed==5
    assert pause.remaining_s==0 and run(pause)==(None,0,[])


def test_critical_pressure_stops_without_waiting():
    assert run(AdmissionPause(True),pressure=True)==(None,0,[])


@pytest.mark.parametrize('field',['used','sout'])
def test_swap_pressure_refuses_even_when_allocation_predicate_recovers(field):
    pause=AdmissionPause(True); calls=[]
    def swap():
        calls.append(1)
        return NS(**{k:17_000_000 if k==field and len(calls)>1 else 0 for k in ['used','sout']})
    assert pause.wait(lambda:(10,100,110,100),critical=60,reason='test',
        clear_cache=lambda:pytest.fail('no clear allowed'),swap=swap) is None


def test_bad_environment_fails_closed(monkeypatch):
    monkeypatch.setenv('VMODEL_ADMISSION_PAUSE','yes')
    with pytest.raises(ValueError):AdmissionPause.from_environment()

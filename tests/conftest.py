import pytest
from datetime import datetime


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    report.start = call.start
    report.stop = call.stop


def pytest_terminal_summary(terminalreporter):
    terminalreporter.ensure_newline()
    terminalreporter.section('start/stop times', sep='-', bold=True)
    for stat in terminalreporter.stats.values():
        for report in stat:
            if getattr(report, 'when', None) == 'call':
                start = datetime.fromtimestamp(report.start)
                stop = datetime.fromtimestamp(report.stop)
                terminalreporter.write_line(f'{report.nodeid:20}: {start:%Y-%m-%d %H:%M:%S} - {stop:%Y-%m-%d %H:%M:%S}')

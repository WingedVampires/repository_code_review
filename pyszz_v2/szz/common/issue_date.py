import logging as log
from typing import Dict
from typing import Set

import dateparser as dp
import datetime
from git import Commit


class IssueDateInfo():
    def __init__(self, source: str, parsed: 'datetime', date_tag: str):
        self.source = source
        self.parsed = parsed
        self.date_tag = date_tag

    def __repr__(self):
        return f"{self.__class__.__name__}(source_date={self.source},parsed_date={self.parsed},date_tag={self.date_tag}"


def parse_issue_date(commit: Dict) -> 'IssueDateInfo':
    """
    Reads iso date from commit and returns a MyIssueDate.
    Note: if the date is not timezone aware, it is assumed to be UTC.
    """

    source_date = None
    date_tag = None
    if 'earliest_issue_date' in commit:
        source_date = commit['earliest_issue_date']
        date_tag = 'earliest_issue_date'
    elif 'best_scenario_issue_date' in commit:
        source_date = commit['best_scenario_issue_date']
        date_tag = 'best_scenario_issue_date'

    assert source_date is not None, f'No issue date found in commit {commit}'
    assert date_tag is not None, f'Invalid date tag for commit {commit}'

    try:
        # 使用 dateparser 解析日期，但捕获可能发生的异常
        parsed_date = dp.parse(source_date)
        if not parsed_date.tzinfo:
            parsed_date = dp.parse(source_date + ' UTC')
    except AttributeError:
        # 如果出现 ZoneInfo 对象的 localize 错误，使用替代方案
        log.warning(f"dateparser failed to parse '{source_date}', using fallback parsing method")
        try:
            # 尝试使用 datetime 标准库直接解析
            parsed_date = datetime.datetime.fromisoformat(source_date.replace('Z', '+00:00'))
        except ValueError:
            # 如果仍然失败，尝试简单格式
            try:
                parsed_date = datetime.datetime.strptime(source_date, "%Y-%m-%dT%H:%M:%S")
                # 添加 UTC 时区信息
                parsed_date = parsed_date.replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                # 最后的尝试，使用当前时间作为后备，但记录警告
                log.error(f"Could not parse date: {source_date}")
                parsed_date = datetime.datetime.now(datetime.timezone.utc)

    return IssueDateInfo(source_date, parsed_date, date_tag)


def filter_by_date(bic: Set[Commit], issue_date: 'IssueDateInfo') -> Set[Commit]:
    """ Filter commits by authored_date using timestamp of issue date (UTC) """

    bic_new = {commit for commit in bic if commit.authored_date < issue_date.parsed.timestamp()}
    log.info(f'Filtering by issue date returned {len(bic_new)} out of {len(bic)}')

    return bic_new

import re


_HEADING_RE = re.compile(r"^#{1,6}\s+")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")


def format_audit_for_telegram(text: str) -> str:
    """Turn Markdown audit output into plain text that Telegram can display."""
    lines = text.replace("\r\n", "\n").split("\n")
    rendered: list[str] = []
    index = 0

    while index < len(lines):
        if _looks_like_table_row(lines[index]):
            table_lines: list[str] = []
            while index < len(lines) and _looks_like_table_row(lines[index]):
                table_lines.append(lines[index])
                index += 1
            rendered.extend(_render_table(table_lines))
            continue

        rendered.append(_render_plain_line(lines[index]))
        index += 1

    return _collapse_blank_lines("\n".join(rendered)).strip()


def _looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    nonempty = [cell for cell in cells if cell]
    if not nonempty:
        return False
    return all(re.fullmatch(r":?-+:?", cell.replace(" ", "")) for cell in nonempty)


def _render_table(table_lines: list[str]) -> list[str]:
    rows = [_split_row(line) for line in table_lines]
    rows = [row for row in rows if row and not _is_separator_row(row)]
    if not rows:
        return []

    headers = [_clean_inline(cell) for cell in rows[0]]
    header_keys = [header.casefold() for header in headers]
    body = rows[1:] if _looks_like_header(header_keys) else rows
    if body is rows:
        headers = []
        header_keys = []

    if _is_criteria_table(header_keys):
        return _render_criteria_rows(headers, body)
    if _is_tasks_table(header_keys):
        return _render_task_rows(headers, body)
    return _render_generic_rows(headers, body)


def _looks_like_header(header_keys: list[str]) -> bool:
    joined = " ".join(header_keys)
    return any(
        marker in joined
        for marker in ("критерий", "статус", "балл", "действие", "ответствен", "срок")
    )


def _is_criteria_table(header_keys: list[str]) -> bool:
    return "критерий" in header_keys or (
        "статус" in header_keys and "балл" in header_keys
    )


def _is_tasks_table(header_keys: list[str]) -> bool:
    return "действие" in header_keys


def _cell(row: list[str], headers: list[str], *names: str, default: str = "") -> str:
    lowered = [header.casefold() for header in headers]
    for name in names:
        for index, header in enumerate(lowered):
            if name in header and index < len(row):
                return _clean_inline(row[index])
    return default


def _render_criteria_rows(headers: list[str], body: list[list[str]]) -> list[str]:
    lines: list[str] = []
    for index, row in enumerate(body, start=1):
        number = _cell(row, headers, "№", "номер") or str(index)
        title = _cell(row, headers, "критерий") or (row[1] if len(row) > 1 else row[0])
        status = _cell(row, headers, "статус")
        score = _cell(row, headers, "балл")
        evidence = _cell(row, headers, "доказатель")

        headline = f"{number}. {_clean_inline(title)}"
        details = [part for part in (status, f"({score})" if score else "") if part]
        if details:
            headline = f"{headline} — {' '.join(details)}"
        lines.append(headline)

        if evidence and evidence not in {"—", "-", "–"}:
            lines.append(f"   {evidence}")
        lines.append("")
    return lines


def _render_task_rows(headers: list[str], body: list[list[str]]) -> list[str]:
    lines: list[str] = []
    for row in body:
        action = _cell(row, headers, "действие") or _clean_inline(row[0])
        owner = _cell(row, headers, "ответствен")
        deadline = _cell(row, headers, "срок")
        source = _cell(row, headers, "основание")

        extras = []
        if owner:
            extras.append(f"ответственный: {owner}")
        if deadline:
            extras.append(f"срок: {deadline}")
        headline = f"• {action}"
        if extras:
            headline = f"{headline} ({', '.join(extras)})"
        lines.append(headline)
        if source and source not in {"—", "-", "–"}:
            lines.append(f"   Основание: {source}")
        lines.append("")
    return lines


def _render_generic_rows(headers: list[str], body: list[list[str]]) -> list[str]:
    lines: list[str] = []
    for row in body:
        if headers:
            pairs = []
            for header, value in zip(headers, [_clean_inline(cell) for cell in row]):
                if header and value:
                    pairs.append(f"{header}: {value}")
            if pairs:
                lines.append("• " + "; ".join(pairs))
                lines.append("")
            continue
        values = [_clean_inline(cell) for cell in row if cell]
        if values:
            lines.append("• " + " — ".join(values))
            lines.append("")
    return lines


def _render_plain_line(line: str) -> str:
    return _clean_inline(_HEADING_RE.sub("", line))


def _clean_inline(text: str) -> str:
    cleaned = _BOLD_RE.sub(r"\1", text)
    cleaned = _ITALIC_RE.sub(r"\1", cleaned)
    return cleaned.strip()


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)

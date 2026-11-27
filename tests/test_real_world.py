from human_or_ai.real_world import load_real_world, read_csv, validate

LONG = " ".join(["word"] * 60)


def _write(tmp_path, body: str):
    path = tmp_path / "rw.csv"
    path.write_text("text,label,source,notes\n" + body, encoding="utf-8")
    return path


def test_valid_rows_have_no_issues(tmp_path):
    path = _write(tmp_path, f'"{LONG}",human,own:email,\n"{LONG} ai",ai,chatgpt,paraphrased\n')
    assert validate(read_csv(path)) == []


def test_reports_empty_row_invalid_label_and_short_text(tmp_path):
    path = _write(tmp_path, ',,,\n"short text",robot,own,\n')
    messages = [str(i) for i in validate(read_csv(path))]
    assert any("row 2: error: empty row" in m for m in messages)
    assert any("row 3: error: invalid label" in m for m in messages)
    assert any("row 3: warning: only 2 words" in m for m in messages)


def test_reports_empty_text_and_duplicate(tmp_path):
    path = _write(tmp_path, f',human,own,\n"{LONG}",ai,gemini,\n"{LONG}",ai,gemini,\n')
    messages = [str(i) for i in validate(read_csv(path))]
    assert any("row 2: error: empty text" in m for m in messages)
    assert any("row 4: warning: same text as row 3" in m for m in messages)


def test_load_drops_errors_and_maps_labels(tmp_path):
    path = _write(tmp_path, f'"{LONG}",Human,own,\n"{LONG} x",AI,llama,Paraphrased\n,,,\n"x",bad,own,\n')
    df = load_real_world(path)
    assert list(df["label"]) == [0, 1]
    assert list(df["paraphrased"]) == [False, True]


def test_multiline_text_and_bom(tmp_path):
    path = tmp_path / "rw.csv"
    path.write_text(f'text,label,source,notes\n"line one\nline two {LONG}",human,own,\n', encoding="utf-8-sig")
    df = load_real_world(path)
    assert len(df) == 1 and "\n" in df["text"][0]


def test_template_csv_is_valid():
    assert validate(read_csv()) == []

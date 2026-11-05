from ytviews.drafts import delete_draft, load_drafts, range_sentence, save_draft


def test_drafts_save_load_delete(tmp_path):
    path = tmp_path / "d" / "drafts.json"
    assert load_drafts(path) == {}
    assert save_draft("  My   video ", {"title": "A"}, path) == "My video"
    assert save_draft("", {"title": "Second title"}, path) == "Second title"  # named after it
    drafts = load_drafts(path)
    assert list(drafts) == ["Second title", "My video"]  # newest first
    assert drafts["My video"]["title"] == "A" and "saved_at" in drafts["My video"]
    save_draft("My video", {"title": "B"}, path)  # overwrite
    assert load_drafts(path)["My video"]["title"] == "B" and len(load_drafts(path)) == 2
    assert delete_draft("My video", path) and not delete_draft("nope", path)
    assert list(load_drafts(path)) == ["Second title"]


def test_a_broken_drafts_file_is_no_drafts(tmp_path):
    path = tmp_path / "drafts.json"
    path.write_text("{not json")
    assert load_drafts(path) == {}
    path.write_text("[1, 2]")
    assert load_drafts(path) == {}


def test_range_sentence():
    text = range_sentence("1.2K", "9.8K", 7, "views", coverage=0.79)
    assert "8 in 10" in text and "**1.2K**" in text and "79%" in text
    assert "held out" not in range_sentence("1", "2", 30, "likes", nominal=0.9)
    assert "9 in 10" in range_sentence("1", "2", 30, "likes", nominal=0.9)

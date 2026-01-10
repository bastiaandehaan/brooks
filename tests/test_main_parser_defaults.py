from main import build_parser


def test_main_parser_defaults():
    p = build_parser()
    args = p.parse_args([])

    assert args.symbol == "US500.cash"
    assert args.session_tz == "America/New_York"
    assert args.day_tz == "America/New_York"
    assert args.session_start == "09:30"
    assert args.session_end == "15:00"
    assert args.max_trades_day == 2

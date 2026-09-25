"""
briefing.py
===========
Turn the forecast into plain English.

    %run briefing.py

The other output scripts are built for someone who already knows what a
posterior median or a blend weight is. This one is built for everyone else:
a reader who wants to know who is favored, by how much, which races decide
it, and how much to trust the answer.

Three principles guide the wording, because a forecast that is misread is
worse than no forecast at all:

1. A probability is not a prediction. The single most common misreading of
   an election forecast is treating "85% chance" as "will happen". Every
   probability here is therefore paired with a frequency ("about 15 times in
   100 the other side wins") and, where it helps, an everyday comparison.

2. Seat counts get ranges, not points. "241 seats" invites false precision.
   "Most likely around 241, and the model would not be surprised by anything
   from 222 to 261" is what the simulations actually say.

3. Uncertainty is stated, not buried. The closing section names what this
   model does not know, including where its own inputs are thin. A reader
   who finishes the briefing should be able to say why the number might be
   wrong, not just what the number is.

Reads:  outputs/chamber_control.json, outputs/race_probabilities.csv
Writes: outputs/forecast_briefing.md  (and prints the same text)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

FOLDER = "midterms-2026"
try:
    PROJECT = Path(__file__).resolve().parent
except NameError:                                     # pasted into a notebook cell
    here = Path.cwd().resolve()
    PROJECT = next((c for base in [here, *here.parents]
                    for c in (base, base / FOLDER, base / "research-papers" / FOLDER)
                    if (c / "config.py").is_file() and (c / "run_pipeline.py").is_file()), None)
    if PROJECT is None:
        raise SystemExit("Could not find the midterms-2026 folder. Run START_HERE.py first.")

sys.path.insert(0, str(PROJECT))
import config                                          # noqa: E402

OUT = PROJECT / "outputs"
CC = OUT / "chamber_control.json"
RP = OUT / "race_probabilities.csv"

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}


# --------------------------------------------------------------------------
# Plain-English translation helpers
# --------------------------------------------------------------------------
def favor_phrase(p: float, party: str = "Democrats", other: str = "Republicans") -> str:
    """Turn a probability into words a non-specialist reads the same way a
    statistician reads the number. The bands follow the convention most
    outlets use, which keeps 'toss-up' genuinely close rather than a label
    for anything under a coin flip."""
    if p >= 0.97: return f"{party} are all but certain to win"
    if p >= 0.90: return f"{party} are very heavy favorites"
    if p >= 0.75: return f"{party} are clear favorites"
    if p >= 0.60: return f"{party} are favored"
    if p >= 0.55: return f"{party} have a slight edge"
    if p > 0.45:  return "this is essentially a coin flip"
    return favor_phrase(1 - p, other, party)


def odds_sentence(p: float) -> str:
    """Pair every probability with the frequency it implies. A reader who
    sees only '38%' tends to round it to 'no'; '38 times in 100' does not
    round as easily."""
    pct = round(p * 100)
    if pct >= 99:  return "The model gives this better than 99 chances in 100."
    if pct <= 1:   return "The model gives this less than 1 chance in 100."
    return (f"The model gives this {pct} chances in 100, which also means the other "
            f"outcome happens {100 - pct} times in 100.")


def margin_words(m: float) -> str:
    """A margin is a lead, so say it as a lead. The convention throughout the
    project is Democratic share minus Republican share in percentage points."""
    side = "Democratic" if m >= 0 else "Republican"
    a = abs(m)
    if a < 1:  size = "a lead too small to call, well under a point"
    elif a < 3: size = f"a narrow {a:.1f} point lead"
    elif a < 8: size = f"a {a:.1f} point lead"
    elif a < 15: size = f"a comfortable {a:.1f} point lead"
    else: size = f"a {a:.0f} point lead, not competitive"
    return f"{side}: {size}"


def race_name(row) -> str:
    """H-PA-07 -> 'Pennsylvania 7th district'. S-OH-special -> 'Ohio Senate
    (special election)'."""
    parts = str(row.race_id).split("-")
    state = STATE_NAMES.get(parts[1], parts[1]) if len(parts) > 1 else row.race_id
    if row.office == "House":
        d = parts[2] if len(parts) > 2 else "?"
        if d == "01" and config.HOUSE_SEATS_BY_STATE.get(parts[1], 0) == 1:
            return f"{state} at-large"
        n = int(d) if d.isdigit() else d
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n if isinstance(n, int) and n % 100 not in (11, 12, 13) else 0, "th")
        return f"{state} {n}{suffix} district"
    special = " (special election)" if "special" in parts else ""
    return f"{state} {'Senate' if row.office == 'Senate' else 'Governor'}{special}"


def tipping_point(senate: pd.DataFrame) -> tuple[str, float] | None:
    """Which race hands over the majority?

    Order every contested seat from most to least likely Democratic and walk
    down the list adding seats to the 34 Democrats already hold and are not
    defending this year. The race that carries the running total past 50 is
    the tipping point: the closest thing to a single race that decides the
    chamber. It is the race worth watching on election night, and it is
    usually more informative than the chamber probability itself, because it
    names the specific contest the majority runs through.
    """
    held = config.SENATE_SEATS_NOT_UP["D"]
    need = config.SENATE_MAJORITY
    ranked = senate.sort_values("p_dem_caucus", ascending=False)
    total = held
    for _, row in ranked.iterrows():
        total += 1
        if total >= need:
            return race_name(row), float(row.p_dem_caucus)
    return None


# --------------------------------------------------------------------------
# Build the briefing
# --------------------------------------------------------------------------
def main() -> None:
    if not CC.exists():
        raise SystemExit(f"No forecast yet: {CC} does not exist.\n"
                         "Run:  %run run_pipeline.py --from 09")
    d = json.loads(CC.read_text())
    r = pd.read_csv(RP) if RP.exists() else pd.DataFrame()
    L: list[str] = []
    add = L.append

    add(f"# 2026 Midterms: what the model says")
    add("")
    add(f"*As of {d['asof']}, {d['days_to_election']} days before the election. "
        f"Built from {d['n_sims']:,} simulated elections.*")
    add("")

    if d["provenance"] == "fixture":
        add("> **Read no further as a forecast.** Some inputs in this run are "
            "placeholders, not real data, so the numbers below test that the "
            "pipeline works. They do not describe the election. Run "
            "`check_data.py` to see which inputs are missing.")
        add("")

    # ---------------- the one-paragraph answer ----------------
    add("## The short version")
    add("")
    h, s, g = d["House"], d["Senate"], d["Governor"]
    add(f"**House of Representatives.** {favor_phrase(h['p_dem_control'])}. "
        f"{odds_sentence(h['p_dem_control'])} The most likely result is about "
        f"{h['dem_seats_median']:.0f} Democratic seats out of 435, and the model would not "
        f"be surprised by anything between {h['dem_seats_p10']:.0f} and {h['dem_seats_p90']:.0f}. "
        f"A party needs {h['majority_threshold']} for control.")
    add("")
    tie = s.get("p_dem_50_seats_tie")
    tie_note = ""
    if tie:
        tie_note = (f" Note that a 50-50 Senate is a real possibility, at {tie:.0%}, and a "
                    f"50-50 Senate is controlled by whichever party holds the vice presidency, "
                    f"which is currently the Republicans. That is why the Democrats need "
                    f"{s['majority_threshold']} seats and not 50.")
    add(f"**Senate.** {favor_phrase(s['p_dem_control'])}. "
        f"{odds_sentence(s['p_dem_control'])} The most likely result is about "
        f"{s['dem_seats_median']:.0f} Democratic seats out of 100, with a plausible range of "
        f"{s['dem_seats_p10']:.0f} to {s['dem_seats_p90']:.0f}.{tie_note}")
    add("")
    add(f"**Governors.** {favor_phrase(g['p_dem_control'])} to hold a majority of "
        f"the 50 governorships. {odds_sentence(g['p_dem_control'])} The most likely result is "
        f"about {g['dem_seats_median']:.0f} Democratic governors, with a range of "
        f"{g['dem_seats_p10']:.0f} to {g['dem_seats_p90']:.0f}. Only 36 of the 50 seats are "
        f"on the ballot this year, so most of that total is already settled.")
    add("")

    # ---------------- how to read a probability ----------------
    add("## How to read these numbers")
    add("")
    add("A forecast is a weather report, not a verdict. When a forecaster says "
        "70% chance of rain, they are not promising rain. They are saying that on "
        "days that look like this one, it rains about 7 times in 10, and stays dry "
        "about 3 times in 10. A good forecaster's 70% days are dry 30% of the time; "
        "if they never were, the forecaster would be underconfident.")
    add("")
    add("Election forecasts work the same way, with one extra wrinkle: there is only "
        "one election, so you never get to watch the other 99 runs. That is what the "
        "simulations are for. The model builds the election "
        f"{d['n_sims']:,} times, each time drawing a different plausible national "
        "environment and a different plausible polling error, and counts how often "
        "each side ends up with a majority.")
    add("")
    add("Two habits protect you from misreading it:")
    add("")
    add("1. Read the range, not the point. \"241 seats\" is the middle of a distribution, "
        "not a prediction. The range is the forecast.")
    add("2. An upset is not a failed forecast. If the trailing side wins a race the model "
        "gave them 30%, the model was not wrong. Roughly a third of such races are "
        "supposed to go that way.")
    add("")

    if len(r):
        # ---------------- the races that decide it ----------------
        add("## The races that decide the Senate")
        add("")
        sen = r[r.office == "Senate"].copy()
        tp = tipping_point(sen)
        if tp:
            add(f"**The tipping point is {tp[0]}.** Line every Senate race up from safest "
                f"Democratic to safest Republican and walk down the list: this is the race "
                f"where the Democrats' running total crosses "
                f"{config.SENATE_MAJORITY} seats. Whoever wins it very probably wins the "
                f"chamber. The model puts the Democratic side of that race at {tp[1]:.0%}.")
            add("")
        close = sen[(sen.p_dem_caucus > 0.15) & (sen.p_dem_caucus < 0.85)].sort_values(
            "p_dem_caucus", ascending=False)
        if len(close):
            add(f"The {len(close)} genuinely contested Senate races, most to least Democratic:")
            add("")
            add("| Race | Democratic chance | Current polling/model margin | Polls? |")
            add("| --- | --- | --- | --- |")
            for _, row in close.iterrows():
                polled = "yes" if row.polled else "no polls yet"
                add(f"| {race_name(row)} | {row.p_dem_caucus:.0%} | "
                    f"{margin_words(row.blend_margin)} | {polled} |")
            add("")

        # ---------------- house ----------------
        add("## The House, in plain terms")
        add("")
        hr = r[r.office == "House"].copy()
        safe_d = int((hr.p_dem >= 0.95).sum()); safe_r = int((hr.p_dem <= 0.05).sum())
        comp = hr[(hr.p_dem > 0.25) & (hr.p_dem < 0.75)]
        add(f"Of 435 districts, {safe_d} are effectively locked for the Democrats and "
            f"{safe_r} for the Republicans. That leaves {435 - safe_d - safe_r} in play, of "
            f"which {len(comp)} are close enough to be real toss-ups. The House majority is "
            f"decided by that last group, which is why a national swing of two or three "
            f"points moves the seat total so much.")
        add("")
        lean = hr[(hr.p_dem > 0.35) & (hr.p_dem < 0.65)].sort_values("p_dem", ascending=False)
        if len(lean):
            add(f"The {len(lean)} closest districts:")
            add("")
            add("| District | Democratic chance | Margin | Polls? |")
            add("| --- | --- | --- | --- |")
            for _, row in lean.head(30).iterrows():
                add(f"| {race_name(row)} | {row.p_dem:.0%} | {margin_words(row.blend_margin)} | "
                    f"{'yes' if row.polled else 'no polls yet'} |")
            if len(lean) > 30:
                add(f"")
                add(f"*({len(lean) - 30} more in `outputs/race_probabilities.csv`.)*")
            add("")

        # ---------------- what would have to happen ----------------
        add("## What would have to change the answer")
        add("")
        n_polled = int(r.polled.sum())
        add(f"The forecast leans on two things: polls, where they exist ({n_polled} of "
            f"{len(r)} races have at least one), and the historical relationship between "
            f"the national mood and seat outcomes everywhere else.")
        add("")
        add("For the underdog to win each chamber:")
        add("")
        add(f"- **House.** Republicans need the national environment to move roughly "
            f"{abs(h['dem_seats_median'] - h['majority_threshold']) / 8:.0f} to "
            f"{abs(h['dem_seats_median'] - h['majority_threshold']) / 5:.0f} points in their "
            f"direction between now and November, or the polls to be wrong in their favor by "
            f"about that much. Both happen, but rarely together.")
        add(f"- **Senate.** Democrats are defending a map where the seats they need are in "
            f"states that lean Republican. They can win the popular vote for the Senate "
            f"comfortably and still fall short, because seats, not votes, are what count.")
        add(f"- **Governors.** Most governorships are not up this year, so even a very good "
            f"night moves the national total by only a few seats.")
        add("")

        # ---------------- honesty ----------------
        add("## What this model does not know")
        add("")
        add("Every forecast has blind spots. These are this one's, stated plainly so you can "
            "discount the numbers yourself rather than trusting them wholesale.")
        add("")
        unpolled_close = int(((~r.polled) & (r.p_dem > 0.3) & (r.p_dem < 0.7)).sum())
        add(f"- **Thin polling below the statewide level.** House district polling is scarce and "
            f"unevenly distributed. {unpolled_close} competitive races currently have no poll at "
            f"all, so their forecast comes from district partisanship and the national "
            f"environment rather than from anyone asking voters there.")
        add("- **Polls have been wrong in the same direction twice recently.** 2016 and 2020 both "
            "understated Republican support in similar states. The model builds in a correlated "
            "polling error for exactly this, which is why its ranges are wide, but a third miss "
            "of that kind would still push results toward the Republican end of every range.")
        add("- **Redistricting is unsettled.** Several states are litigating their maps. Where a "
            "district is new or redrawn, the model deliberately widens its uncertainty, because "
            "past results in the old district say less about the new one.")
        add("- **Late-breaking events are not forecastable.** A scandal, an economic shock or an "
            "external crisis in October is not in these numbers, and cannot be.")
        add("- **Turnout in midterms is volatile.** Roughly 40% of eligible voters show up, "
            "against 60%+ in presidential years, and which 40% shows up moves results more "
            "than persuasion does.")
        add("")
        add("---")
        add("")
        add(f"*Full numbers for all {len(r)} races: `outputs/race_probabilities.csv`. "
            f"Data quality label for this run: **{d['provenance']}**.*")

    text = "\n".join(L)
    dest = OUT / "forecast_briefing.md"
    dest.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n\n[saved to {dest}]")


if __name__ == "__main__":
    main()

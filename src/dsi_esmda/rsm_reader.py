"""
rsm_reader.py
=============

Read an Eclipse / OPM Flow ".RSM" summary file into a pandas DataFrame,
and optionally save it as a CSV file.

WHAT IS AN RSM FILE?
--------------------
It is a plain text report. Inside it, the results are printed in several
"pages", and each page starts with a line like:

    SUMMARY OF RUN Model1

Each page (we will call it a *block*) looks like this:

        SUMMARY OF RUN Model1        <- block marker
        TIME     YEARS   FOPR   WOPR   WOPR      <- line 1: keyword names
        DAYS     YEARS   SM3/DAY SM3/DAY SM3/DAY <- line 2: units
                         *10**3                  <- line 3: multiplier (OPTIONAL!)
                                PROD1  PROD2     <- line 4: well names (may be blank)
                                                 <- line 5: blank separator
        0        0       0      0      0         <- data rows start here
        30.0     0.082   636.0  182.3  140.6
        ...

Every block has the same TIME column, but different result columns.
So to get one nice table we read every block and glue them side by side.

HOW THIS FILE IS ORGANISED (the "OOP" part)
-------------------------------------------
There are two classes:

    RSMBlock  -> knows how to understand ONE block ("one page")
    RSMFile   -> knows how to split a file into blocks and glue them together

An RSMFile *has many* RSMBlock objects. This is called **composition**:
a big object built out of smaller objects. It keeps each class small and
easy to read.

QUICK USE
---------
    from dsi_esmda import RSMFile

    rsm = RSMFile("EGG-M1.RSM")     # step 1: create the object
    df = rsm.read()                 # step 2: get the pandas DataFrame
    rsm.to_csv()                    # step 3: write EGG-M1.csv next to it

Or, if you just want one line:

    from dsi_esmda import read_rsm
    df = read_rsm("EGG-M1.RSM")

From the command line:

    python -m dsi_esmda.rsm_reader EGG-M1.RSM
"""

# ---------------------------------------------------------------------------
# Imports: extra tools we borrow from Python and from other libraries.
# ---------------------------------------------------------------------------
import re                 # "regular expressions": searching for text patterns
from pathlib import Path  # a friendly way to work with file paths

import numpy as np        # fast arrays, and float comparison with a tolerance
import pandas as pd       # the table / DataFrame library


# This pattern finds a multiplier cell such as "*10**3" or "*10**6"
# and captures the number after the stars (the 3 or the 6).
_MULTIPLIER_PATTERN = re.compile(r"\*10\*\*(-?\d+)")

# How close two report times must be to count as the same time, in days.
# Report times are printed to six decimals, so a value stored as 359.999999
# must still match 360. Comparing floats with "==" is the mistake this
# constant exists to prevent.
_TIME_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# A few small helper functions.
# A function that starts with "_" is a hint to other programmers:
# "this is an internal detail, you do not need to call it yourself".
# ---------------------------------------------------------------------------
def _split_cells(line):
    """Cut one line of the RSM file into a list of clean text cells.

    RSM files separate columns with TAB characters ("\\t").
    Both header lines and data lines begin with a TAB, so the very first
    piece is always empty and we throw it away.

    Example:
        "\\t 30.00000\\t    0.082136\\t"  ->  ["30.00000", "0.082136", ""]
    """
    pieces = line.split("\t")     # cut the line wherever there is a TAB
    pieces = pieces[1:]           # drop the empty piece before the first TAB
    return [piece.strip() for piece in pieces]   # remove padding spaces


def _drop_trailing_blanks(cells):
    """Remove the empty cells a line's trailing TABs leave behind.

    A line ending in a TAB produces one extra empty cell, and some writers
    pad every line to a fixed width. Those blanks are not missing values, so
    they must go before we count how many numbers a row really has.
    """
    cells = list(cells)
    while cells and cells[-1] == "":
        cells.pop()
    return cells


def _looks_like_number(text):
    """Return True if this piece of text can be read as a number.

    We use this to tell data rows apart from header rows: a data row
    always starts with a TIME value like "30.00000", while a header row
    starts with a word like "TIME" or with nothing at all.
    """
    try:
        float(text)
        return True
    except ValueError:
        return False


def _multiplier_of(cell):
    """Turn a multiplier cell like "*10**3" into the number 1000.0.

    If the cell is empty (most columns have no multiplier) we return 1.0,
    because multiplying by 1 changes nothing.
    """
    match = _MULTIPLIER_PATTERN.search(cell)
    if match:
        exponent = int(match.group(1))   # the "3" in "*10**3"
        return 10.0 ** exponent
    return 1.0


# ---------------------------------------------------------------------------
# CLASS 1: one block ("one page") of the RSM file
# ---------------------------------------------------------------------------
class RSMBlock:
    """One "SUMMARY OF RUN" table taken from an RSM file.

    A *class* is a blueprint. An *object* is one real thing built from that
    blueprint. Here, the blueprint describes "a block of RSM results", and
    each object holds the numbers and names of one specific block.
    """

    def __init__(self, lines):
        """The constructor: it runs automatically when we create an object.

        `self` means "this particular object". Anything we store on `self`
        is called an *attribute* and stays available afterwards.

        `lines` is the list of text lines belonging to this block,
        starting with the "SUMMARY OF RUN ..." line.
        """
        self.lines = lines          # keep the raw text, useful for debugging

        # These attributes get filled in by _parse() below.
        self.run_name = ""          # e.g. "Model1"
        self.keywords = []          # e.g. ["TIME", "FOPR", "WOPR", ...]
        self.units = []             # e.g. ["DAYS", "SM3/DAY", ...]
        self.wells = []             # e.g. ["", "", "PROD1", ...]
        self.multipliers = []       # e.g. [1.0, 1000.0, 1.0, ...]
        self.values = []            # the data rows, as lists of floats

        self._parse()               # do the real work immediately

    # -- the parsing machinery ---------------------------------------------
    def _parse(self):
        """Read the header lines, then the data rows."""

        # 1) The block marker line, e.g. "\tSUMMARY OF RUN Model1      \t"
        marker = self.lines[0].strip()
        self.run_name = marker.replace("SUMMARY OF RUN", "").strip()

        # 2) Walk through the lines after the marker and sort them into
        #    "header lines" and "data lines". We stop collecting headers as
        #    soon as we meet a line whose first cell is a number.
        header_lines = []
        data_lines = []
        for line in self.lines[1:]:
            if not line.strip():        # completely empty line -> ignore
                continue
            cells = _split_cells(line)
            if cells and _looks_like_number(cells[0]):
                data_lines.append(cells)
            else:
                header_lines.append(cells)

        if len(header_lines) < 2:
            raise ValueError(
                f"Block '{self.run_name}' has no usable header lines."
            )

        # 3) The FIRST header line always holds the keyword names.
        #    Its length tells us how many columns this block really has.
        #    (Careful: most blocks have 10 columns, but the last block of a
        #     file can be shorter, so we must never hard-code 10.)
        self.keywords = header_lines[0]
        n_columns = len(self.keywords)

        # 4) The SECOND header line always holds the units.
        self.units = self._fit(header_lines[1], n_columns)

        # 5) The remaining header lines are trickier, because their number
        #    changes from block to block:
        #      - a line containing "*10**" is the multiplier line (optional)
        #      - the next line that still has some text is the well names
        #      - fully blank lines are just separators
        #    So we decide by looking at the CONTENT, not at the position.
        multiplier_cells = [""] * n_columns
        well_cells = [""] * n_columns

        for cells in header_lines[2:]:
            cells = self._fit(cells, n_columns)
            has_text = any(cell for cell in cells)
            if not has_text:
                continue                                    # blank separator
            if any("*10**" in cell for cell in cells):
                multiplier_cells = cells                    # multiplier line
            else:
                well_cells = cells                          # well-name line

        self.wells = well_cells
        self.multipliers = [_multiplier_of(cell) for cell in multiplier_cells]

        # 6) A block with headers but no numbers means the run wrote its
        #    report and then stopped. Better to say so here than to hand
        #    back an empty table that looks like a successful read.
        if not data_lines:
            raise ValueError(
                f"Block '{self.run_name}' has header lines but no data rows. "
                "The run probably failed or was stopped before it reported "
                "anything."
            )

        # 7) Finally the numbers. Each data line may carry extra padding
        #    fields at the end, so we drop trailing blanks, check the row is
        #    complete, and turn the first `n_columns` cells into floats.
        for row_index, cells in enumerate(data_lines):
            cells = _drop_trailing_blanks(cells)

            if len(cells) < n_columns:
                raise ValueError(
                    f"Block '{self.run_name}', data row {row_index + 1}: "
                    f"expected {n_columns} values but found {len(cells)} "
                    f"({cells}). The file is probably truncated - a run that "
                    "was killed while writing its report ends exactly like "
                    "this."
                )

            try:
                row = [float(cell) for cell in cells[:n_columns]]
            except ValueError as error:
                raise ValueError(
                    f"Block '{self.run_name}', data row {row_index + 1}: "
                    f"could not read a number from {cells[:n_columns]}."
                ) from error

            # Apply the multiplier, e.g. a value printed as 12.95 with
            # "*10**3" above it really means 12950.0. Forgetting this step
            # silently gives wrong cumulative volumes (FOPT, FWPT, ...).
            row = [value * factor
                   for value, factor in zip(row, self.multipliers)]
            self.values.append(row)

    @staticmethod
    def _fit(cells, n_columns):
        """Make a list exactly `n_columns` long (pad with "" or cut off).

        Some header lines are shorter than the keyword line because the
        trailing empty cells were not printed. This keeps everything aligned.

        `@staticmethod` means: this helper belongs to the class for tidiness,
        but it does not need `self` because it uses no object attributes.
        """
        cells = list(cells[:n_columns])
        while len(cells) < n_columns:
            cells.append("")
        return cells

    # -- things we can ask a block ----------------------------------------
    def column_names(self):
        """Build readable column names such as "TIME", "FOPR", "WOPR:PROD1".

        Field results (FOPR, FWPT, ...) belong to the whole field, so they
        have no well name. Well results (WOPR, WBHP, ...) are repeated once
        per well, so we glue the well name on with a colon to keep them apart.
        """
        names = []
        for keyword, well in zip(self.keywords, self.wells):
            if well:
                names.append(f"{keyword}:{well}")
            else:
                names.append(keyword)
        return names

    def to_dataframe(self):
        """Turn this single block into a small pandas DataFrame."""
        # dtype=float is explicit: without it, a column whose every printed
        # value happens to be whole would come back as integers, and later
        # arithmetic would round instead of dividing.
        return pd.DataFrame(self.values, columns=self.column_names(),
                            dtype=float)

    def unit_map(self):
        """Return a dictionary {column name: unit}, e.g. {"FOPR": "SM3/DAY"}."""
        return dict(zip(self.column_names(), self.units))

    def times(self):
        """This block's TIME column, as a numpy array."""
        names = self.column_names()
        if "TIME" not in names:
            raise ValueError(
                f"Block '{self.run_name}' has no TIME column. Its columns "
                f"are {names[:6]}."
            )
        position = names.index("TIME")
        return np.asarray([row[position] for row in self.values], dtype=float)

    def __repr__(self):
        """What Python prints when you type the object name in the console.

        Defining this is optional, but it makes debugging much nicer.
        """
        return (f"RSMBlock(run={self.run_name!r}, "
                f"columns={len(self.keywords)}, rows={len(self.values)})")


# ---------------------------------------------------------------------------
# CLASS 2: the whole RSM file
# ---------------------------------------------------------------------------
class RSMFile:
    """An RSM file: many blocks glued into one wide table.

    Typical use:

        rsm = RSMFile("DROGON1.RSM")
        df  = rsm.read()
        rsm.to_csv()
    """

    def __init__(self, path):
        """Store the file path. Nothing is read from disk yet.

        Reading only happens when you call .read(). This is deliberate:
        creating the object stays instant, and you decide when to do the work.
        """
        self.path = Path(path)      # Path() gives us handy tools like .stem
        self.blocks = []            # will hold RSMBlock objects
        self.data = None            # will hold the final DataFrame
        self.encoding = None        # which encoding the file turned out to be
        self._units = {}            # {column name: unit}

    # -- step 0: get the text off disk ------------------------------------
    def _read_text(self):
        """Read the file as text, coping with either common encoding.

        Eclipse normally writes plain ASCII, but a deck with accented well
        names, or a file that has been through a CMG or Petrel export, can be
        ISO-8859-1 ("Latin-1"). Decoding those bytes as UTF-8 fails.

        The tempting fix is errors="replace", which never fails - but it
        quietly puts a U+FFFD character into the middle of a WELL NAME, so
        the column ends up called something you can never match in a config,
        and nothing warns you. Instead we try UTF-8 and fall back to
        Latin-1, which can decode any byte sequence at all.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Cannot find the file: {self.path}")

        raw = self.path.read_bytes()
        try:
            text = raw.decode("utf-8")
            self.encoding = "utf-8"
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
            self.encoding = "latin-1"
        return text

    # -- step 1: cut the file into blocks ---------------------------------
    def _read_blocks(self):
        """Split the text file into blocks, one per "SUMMARY OF RUN" line."""
        text = self._read_text()

        blocks = []
        current = None
        # splitlines() handles \n, \r\n and a lone \r alike, so Windows and
        # Unix files both work with no special case.
        for line in text.splitlines():
            if "SUMMARY OF RUN" in line:
                if current:                 # the previous block is finished
                    blocks.append(RSMBlock(current))
                current = [line]            # start collecting a new block
            elif current is not None:
                current.append(line)
        if current:                         # do not forget the last block
            blocks.append(RSMBlock(current))

        if not blocks:
            raise ValueError(
                f"No 'SUMMARY OF RUN' block found in {self.path.name}. "
                "Is this really an RSM file?"
            )
        return blocks

    # -- step 2: glue the blocks together ---------------------------------
    def read(self):
        """Read the file and return one wide pandas DataFrame.

        The result has TIME as its first column, then every result column
        found anywhere in the file.
        """
        self.blocks = self._read_blocks()
        self._check_times_agree()

        # Turn every block into a small table, then glue them side by side.
        # TIME appears in every block, so we keep it only from the first one;
        # `already_used` remembers which column names we have taken already.
        pieces = []
        already_used = set()
        units = {}

        for block in self.blocks:
            piece = block.to_dataframe()
            block_units = block.unit_map()

            new_names = [name for name in piece.columns
                         if name not in already_used]
            already_used.update(new_names)
            for name in new_names:
                units[name] = block_units[name]

            pieces.append(piece[new_names])

        # pd.concat(..., axis=1) puts the small tables next to each other.
        # axis=1 means "join columns"; axis=0 would stack rows instead.
        # Building the list first and concatenating ONCE matters: inserting
        # columns one at a time into a growing DataFrame is quadratic, and
        # pandas warns about the fragmentation it causes.
        self.data = pd.concat(pieces, axis=1)
        self._units = units
        return self.data

    def _check_times_agree(self):
        """Refuse to glue blocks that describe different time steps.

        The blocks are joined SIDE BY SIDE, which is only meaningful if row i
        of every block is the same moment in time. Comparing row COUNTS is
        not enough: two blocks can have the same number of rows and different
        times, and gluing those pairs an oil rate from day 30 with a pressure
        from day 45 - with nothing to show that it happened.
        """
        counts = {len(block.values) for block in self.blocks}
        if len(counts) != 1:
            raise ValueError(
                f"The blocks in {self.path.name} have different numbers of "
                f"rows ({sorted(counts)}). The file is incomplete - the run "
                "was probably still writing when it stopped."
            )

        reference = self.blocks[0].times()
        for block in self.blocks[1:]:
            times = block.times()
            if not np.allclose(times, reference, rtol=0.0,
                               atol=_TIME_TOLERANCE):
                first_bad = int(np.argmax(np.abs(times - reference)
                                          > _TIME_TOLERANCE))
                raise ValueError(
                    f"In {self.path.name}, block '{block.run_name}' reports "
                    f"different times from block "
                    f"'{self.blocks[0].run_name}': at row {first_bad} one "
                    f"says TIME = {reference[first_bad]:g} and the other "
                    f"{times[first_bad]:g}. The blocks cannot be joined."
                )

    # -- step 3: save it --------------------------------------------------
    def to_csv(self, csv_path=None, index=False):
        """Write the table to a CSV file and return the path used.

        By default the CSV is placed next to the RSM file with the same name:
        "DROGON1.RSM" -> "DROGON1.csv".
        """
        if self.data is None:       # be forgiving: read the file if needed
            self.read()

        if csv_path is None:
            csv_path = self.path.with_suffix(".csv")
        csv_path = Path(csv_path)

        self.data.to_csv(csv_path, index=index)
        return csv_path

    # -- convenient extras -------------------------------------------------
    # Each of these reads the file if you have not already. Returning an
    # empty list from `rsm.wells` just because `.read()` had not been called
    # looks like "this file has no wells", which is a bad way to be wrong.
    @property
    def keywords(self):
        """The sorted list of result keywords found in the file.

        `@property` lets us write `rsm.keywords` instead of `rsm.keywords()`.
        It behaves like an attribute but is calculated on demand.
        """
        if self.data is None:
            self.read()
        found = set()
        for block in self.blocks:
            found.update(block.keywords)
        return sorted(found)

    @property
    def wells(self):
        """The sorted list of well names found in the file."""
        if self.data is None:
            self.read()
        found = set()
        for block in self.blocks:
            found.update(well for well in block.wells if well)
        return sorted(found)

    @property
    def units(self):
        """{column name: unit}, e.g. {"WOPR:PROD1": "SM3/DAY"}.

        An .RSM file states its own units, which is why they are worth
        keeping: they become the y-axis labels on the plots.
        """
        if self.data is None:
            self.read()
        return self._units

    @property
    def times(self):
        """The report times of the file, as a numpy array."""
        if self.data is None:
            self.read()
        return self.data["TIME"].to_numpy(dtype=float)

    @property
    def n_blocks(self):
        """How many "SUMMARY OF RUN" pages the file contains."""
        if self.data is None:
            self.read()
        return len(self.blocks)

    def at_times(self, times, tolerance=_TIME_TOLERANCE):
        """Return only the rows whose TIME matches one of `times`.

            rsm.at_times([0, 360, 720])

        The comparison uses a TOLERANCE rather than exact equality. Report
        times are floats printed to six decimals, so a time stored as
        359.9999994 is not `== 360`, and comparing with `isin` would return
        an empty table and no error at all.

        Widen the tolerance when your list of times is rounded:

            rsm.at_times([224, 449], tolerance=1.0)
        """
        if self.data is None:
            self.read()

        available = self.data["TIME"].to_numpy(dtype=float)
        wanted = np.asarray([float(time) for time in times], dtype=float)

        keep = np.zeros(available.shape, dtype=bool)
        for time in wanted:
            keep |= np.abs(available - time) <= float(tolerance)
        return self.data[keep]

    def __repr__(self):
        rows = "not read yet" if self.data is None else f"{len(self.data)} rows"
        return f"RSMFile({self.path.name!r}, {rows})"


# ---------------------------------------------------------------------------
# One-line shortcuts for when you do not care about the objects.
# ---------------------------------------------------------------------------
def read_rsm(path):
    """Read an RSM file and return the pandas DataFrame directly."""
    return RSMFile(path).read()


def rsm_to_csv(path, csv_path=None):
    """Read an RSM file and immediately write it to CSV. Returns the CSV path."""
    return RSMFile(path).to_csv(csv_path)


# ---------------------------------------------------------------------------
# This part only runs when you execute the module directly:
#
#     python -m dsi_esmda.rsm_reader EGG-M1.RSM
#
# Note the "-m" and the dots. Now that this file lives inside a package,
# "python src/dsi_esmda/rsm_reader.py" is the wrong way to run it: Python
# would treat the file as a lone script and would not know the package it
# belongs to. "-m" means "run this module, inside its package".
#
# None of this runs on "from dsi_esmda import RSMFile". That is what the
# `if __name__ == "__main__":` line is for.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m dsi_esmda.rsm_reader <file.RSM> [more.RSM ...]")
        sys.exit(1)

    for file_name in sys.argv[1:]:
        rsm = RSMFile(file_name)
        df = rsm.read()
        csv_path = rsm.to_csv()

        print(f"\n{rsm.path.name}  [{rsm.encoding}]")
        print(f"  blocks : {len(rsm.blocks)}")
        print(f"  rows   : {len(df)}")
        print(f"  columns: {len(df.columns)}")
        print(f"  times  : {rsm.times.min():g} .. {rsm.times.max():g}")
        print(f"  wells  : {', '.join(rsm.wells) if rsm.wells else '(none)'}")
        print(f"  saved  : {csv_path}")
        # Show only the first few columns, otherwise 200+ columns flood
        # the screen and you cannot read anything.
        print(df.iloc[:3, :6].to_string(index=False))
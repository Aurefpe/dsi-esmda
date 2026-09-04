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
    from rsm_reader import RSMFile

    rsm = RSMFile("EGG-M1.RSM")     # step 1: create the object
    df = rsm.read()                 # step 2: get the pandas DataFrame
    rsm.to_csv()                    # step 3: write EGG-M1.csv next to it

Or, if you just want one line:

    from rsm_reader import read_rsm
    df = read_rsm("EGG-M1.RSM")
"""

# ---------------------------------------------------------------------------
# Imports: extra tools we borrow from Python and from other libraries.
# ---------------------------------------------------------------------------
import re                 # "regular expressions": searching for text patterns
from pathlib import Path  # a friendly way to work with file paths

import pandas as pd       # the table / DataFrame library


# This pattern finds a multiplier cell such as "*10**3" or "*10**6"
# and captures the number after the stars (the 3 or the 6).
_MULTIPLIER_PATTERN = re.compile(r"\*10\*\*(-?\d+)")


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

        # 6) Finally the numbers. Each data line may carry one extra
        #    padding field at the end, so we keep only the first
        #    `n_columns` cells and turn them into floats.
        for cells in data_lines:
            row = [float(cell) for cell in cells[:n_columns]]
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
        return pd.DataFrame(self.values, columns=self.column_names())

    def unit_map(self):
        """Return a dictionary {column name: unit}, e.g. {"FOPR": "SM3/DAY"}."""
        return dict(zip(self.column_names(), self.units))

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
        self.units = {}             # {column name: unit}

    # -- step 1: cut the file into blocks ---------------------------------
    def _read_blocks(self):
        """Split the text file into blocks, one per "SUMMARY OF RUN" line."""
        if not self.path.exists():
            raise FileNotFoundError(f"Cannot find the file: {self.path}")

        # encoding="utf-8", errors="replace" makes sure a stray odd character
        # cannot crash the whole read. Python also converts the Windows
        # line endings (\r\n) into plain \n for us automatically.
        text = self.path.read_text(encoding="utf-8", errors="replace")

        blocks = []
        current = None
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

        # Check that all blocks describe the same time steps. They should,
        # because they come from the same simulation run.
        row_counts = {len(block.values) for block in self.blocks}
        if len(row_counts) != 1:
            raise ValueError(
                f"The blocks in {self.path.name} have different numbers of "
                f"rows ({sorted(row_counts)}). The file may be incomplete."
            )

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
        self.data = pd.concat(pieces, axis=1)
        self.units = units
        return self.data

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
    @property
    def keywords(self):
        """The sorted list of result keywords found in the file.

        `@property` lets us write `rsm.keywords` instead of `rsm.keywords()`.
        It behaves like an attribute but is calculated on demand.
        """
        found = set()
        for block in self.blocks:
            found.update(block.keywords)
        return sorted(found)

    @property
    def wells(self):
        """The sorted list of well names found in the file."""
        found = set()
        for block in self.blocks:
            found.update(well for well in block.wells if well)
        return sorted(found)

    def at_times(self, times):
        """Return only the rows whose TIME is in the given list.

            rsm.at_times([0, 360, 720])
        """
        if self.data is None:
            self.read()
        return self.data[self.data["TIME"].isin(times)]

    def __repr__(self):
        rows = "not read yet" if self.data is None else f"{len(self.data)} rows"
        return f"RSMFile({self.path.name!r}, {rows})"


# ---------------------------------------------------------------------------
# A one-line shortcut for when you do not care about the objects.
# ---------------------------------------------------------------------------
def read_rsm(path):
    """Read an RSM file and return the pandas DataFrame directly."""
    return RSMFile(path).read()


def rsm_to_csv(path, csv_path=None):
    """Read an RSM file and immediately write it to CSV. Returns the CSV path."""
    return RSMFile(path).to_csv(csv_path)


# ---------------------------------------------------------------------------
# This part only runs when you execute the file directly, for example:
#
#     python rsm_reader.py EGG-M1.RSM
#
# It does NOT run when you do "from rsm_reader import RSMFile" in another
# script. That is what the `if __name__ == "__main__":` line is for.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rsm_reader.py <file.RSM> [more.RSM ...]")
        sys.exit(1)

    for file_name in sys.argv[1:]:
        rsm = RSMFile(file_name)
        df = rsm.read()
        csv_path = rsm.to_csv()

        print(f"\n{rsm.path.name}")
        print(f"  blocks : {len(rsm.blocks)}")
        print(f"  rows   : {len(df)}")
        print(f"  columns: {len(df.columns)}")
        print(f"  wells  : {', '.join(rsm.wells) if rsm.wells else '(none)'}")
        print(f"  saved  : {csv_path}")
        # Show only the first few columns, otherwise 200+ columns flood
        # the screen and you cannot read anything.
        print(df.iloc[:3, :6].to_string(index=False))
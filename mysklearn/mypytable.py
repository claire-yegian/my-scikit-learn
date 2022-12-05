import copy
import csv
import statistics as stats
import tabulate as tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: 1D list of values in the column
        Notes:
            Raise ValueError on invalid col_identifier
        """
        column = []

        try:
            col_idx = self.column_names.index(col_identifier)
        except:
            raise ValueError("invalid col_identifier")

        if include_missing_values:
            for row in self.data:
                column.append(row[col_idx])
        else:
            for row in self.data:
                if row[col_idx] != "NA":
                    column.append(row[col_idx])
        return column

    def add_column(self, header, new_data):
        """Adds a new column to the table
        Args:
            header(str): the header for the new column
            new_data(list): the data to be added to the table, in order
        """
        if header in self.column_names:
            raise Exception("Column already exists. Change name")
        self.column_names.append(header)
        for i in range(len(self.data)):
            self.data[i].append(new_data[i])

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in range(len(self.data)):
            for item in range(len(self.data[row])):
                try:
                    self.data[row][item] = float(self.data[row][item])
                except:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.
        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        new_table = []

        for i in range(len(self.data)):
            if i not in row_indexes_to_drop:
                new_table.append(self.data[i])

        self.data = new_table

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        infile = open(filename, "r")
        reader = csv.reader(infile)

        self.column_names = next(reader)
        self.data = []

        for row in reader:
            self.data.append(row)
        infile.close()

        self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")
        writer = csv.writer(outfile)

        writer.writerow(self.column_names)
        writer.writerows(self.data)
        outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns:
            list of int: list of indexes of duplicate rows found
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        seen = []
        seen_idxs = []

        for row in self.data:
            seen_str = ""
            for name in key_column_names:
                seen_str += str(row[self.column_names.index(name)]) + ", "
            if seen_str in seen:
                seen_idxs.append(self.data.index(row))
            else:
                seen.append(seen_str)

        return seen_idxs

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        new_table = []

        for row in self.data:
            if "NA" not in row:
                new_table.append(row)

        self.data = new_table

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        column = self.get_column(col_name)
        total = 0
        n = 0
        idxs_to_replace = []

        for i in range(len(column)):
            if (column[i] != "NA") and isinstance(column[i], float):
                total += column[i]
                n += 1
            else:
                idxs_to_replace.append(i)
        avg = total/n

        for idx in range(len(idxs_to_replace)):
            self.data[idxs_to_replace[idx]
                      ][self.column_names.index(col_name)] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            minimum: minimum of the column
            maximum: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column
        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "minimum", "maximum", "mid", "avg", "median"]
        Notes:
            Missing values in the columns to compute summary stats for, should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        summary_stats_headers = ["attribute",
                                 "minimum", "maximum", "mid", "avg", "median"]
        summary_stats = []

        for col in range(len(col_names)):
            column = self.get_column(col_names[col])
            if len(column) > 0:
                maximum = 0
                minimum = column[0]
                total = 0
                n = 0
                for item in column:
                    if item != "NA":
                        if item > maximum:
                            maximum = item
                        if item < minimum:
                            minimum = item
                        total += item
                        n += 1
                mid = (minimum + maximum)/2
                avg = total/n

                new_col = []
                for i in range(len(column)):
                    if column[i] != "NA":
                        new_col.append(column[i])
                column = new_col
                column.sort()
                if n % 2 == 0:
                    med_idx = [n//2 - 1, n//2]
                else:
                    med_idx = [n//2]

                for i in range(len(column)//2 + 1):
                    if i in med_idx:
                        if len(med_idx) == 2:
                            med = (column[i] + column[i+1])/2
                            break
                        else:
                            med = column[i]
                            break

                row = [col_names[col], minimum, maximum, mid, avg, med]
                summary_stats.append(row)

        return MyPyTable(summary_stats_headers, summary_stats)

    def make_joined_header(self, table1_cols, table2_cols, key_column_names):
        """Return a new header that combines the headers of the two tables
            being joined such that no column name is repeated
        Args:
            table1_cols(list of str): the header of the first table
            table2_cols(list of str): the header of the second table
            key_column_names(list of str): column names to use as row keys.
        Returns:
            list of str: the joined header
        """
        headers = []
        headers.extend(table1_cols)
        headers_to_add = table2_cols.copy()
        for name in table1_cols:
            if name in table2_cols:
                headers_to_add.remove(name)
        headers.extend(headers_to_add)
        return headers

    def find_match(self, table1_cols, row1, table2_cols, row2, key_column_names):
        """Find if the given rows match for the key column(s)
        Args:
            table1_cols(list of str): the header of the first table
            row1(list): the first row we're comparing
            table2_cols(list of str): the header of the second table
            row2(list): the second row we're comparing
            key_column_names(list of str): column names to use as row keys.
        Returns:
            boolean: true if the rows matched, false otherwise
        """
        match = True
        for name in key_column_names:
            if (name in table1_cols) and (name in table2_cols):
                if row1[table1_cols.index(name)] != row2[table2_cols.index(name)]:
                    match = False
        return match

    def join_rows(self, row1, row2, table2_cols, key_column_names):
        """Join two rows
        Args:
            row1(list): the first row to be joined
            row2(list): the second row to be joined
            table2_cols(list of str): the header of the second table
            key_column_names(list of str): column names to use as row keys.
        Returns:
            list: the joined row
        """
        new_row = []
        row_to_add = row2.copy()
        for name in key_column_names:
            row_to_add.remove(row2[table2_cols.index(name)])
        new_row.extend(row1)
        new_row.extend(row_to_add)
        return new_row

    def fill_missing_vals(self, joined_header, row_header, row_to_fill):
        """Fill missing values in a row with "NA"
        Args:
            joined_header(list of str): the header of the table where we'll add
                the filled row
            row_header(list of str): the header of the row we're filling
            row_to_fill(list): the row that needs missing values filled
        Returns:
            list: the filled row
        """
        new_row = ['NA' for x in range(len(joined_header))]
        for i in range(len(row_header)):
            matched_row_idx = joined_header.index(row_header[i])
            new_row[matched_row_idx] = row_to_fill[i]
        return new_row

    def check_if_in_joined(self, row, row_cols, joined, joined_cols, key_column_names):
        """Check if a given row is in a joined table
        Args:
            row(list): the row to check
            row_cols(list): the header of the row we're checking
            joined(list of lists): the joined table
            joined_cols(list of str): the header of the joined table
            key_column_names(list of str): column names to use as row keys.
        Returns:
            boolean: true if the row was in the table, false if not
        """
        present = True
        for joined_row in joined:
            if self.find_match(row_cols, row, joined_cols, joined_row, key_column_names):
                present = False
        return present

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        joined_table_headers = self.make_joined_header(
            self.column_names, other_table.column_names, key_column_names)
        joined_table = []

        for self_row in self.data:
            for other_row in other_table.data:
                if self.find_match(self.column_names, self_row, other_table.column_names, other_row, key_column_names):
                    joined_table.append(self.join_rows(
                        self_row, other_row, other_table.column_names, key_column_names))

        return MyPyTable(joined_table_headers, joined_table)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """
        joined_header = self.make_joined_header(
            self.column_names, other_table.column_names, key_column_names)
        joined_table = []

        for self_row in self.data:
            matched = False
            for other_row in other_table.data:
                if self.find_match(self.column_names, self_row, other_table.column_names, other_row, key_column_names):
                    matched = True
                    joined_table.append(self.join_rows(
                        self_row, other_row, other_table.column_names, key_column_names))
            if not matched:
                joined_table.append(self.fill_missing_vals(
                    joined_header, self.column_names, self_row))

        for unfilled_row in other_table.data:
            if self.check_if_in_joined(unfilled_row, other_table.column_names, joined_table, joined_header, key_column_names):
                joined_table.append(self.fill_missing_vals(
                    joined_header, other_table.column_names, unfilled_row))

        return MyPyTable(joined_header, joined_table)
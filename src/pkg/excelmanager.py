import os
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import Protection, NamedStyle

class ExcelManager:
    def __init__(self, directory):
        self.directory = directory

    def setup_excel_writer(self, file_name, df_columns):
        if not os.path.exists(file_name):
            # Create a new workbook and setup headers
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Sales"
            # Write headers
            for i, col in enumerate(df_columns, start=1):
                sheet.cell(row=1, column=i, value=str(col))
            workbook.save(file_name)
            workbook.close()
        
        # Return the ExcelWriter object
        writer = pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='overlay')
        return writer

    def apply_formatting_to_all_files(self):
        # Iterate over all files in the directory
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if file.endswith(".xlsx"):
                    file_path = os.path.join(root, file)
                    self.duplicate_sheet_with_zeros(file_path)
                    self.apply_table_formatting_and_adjust_columns(file_path)
                    self.apply_number_format(file_path)
                    self.lock_columns(file_path, "007006")

    def column_number_to_letter(self, n):
        """Convert a column number (e.g., 1) to a column letter (e.g., 'A')."""
        string = ""
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            string = chr(65 + remainder) + string
        return string

    def apply_table_formatting_and_adjust_columns(self, file_name):
        book = load_workbook(file_name)
        sheets = {'Sales': 'Table1',
                  'Sample': 'Table2',
                  'Demo': 'Table3'}
        for sheet, table_name in sheets.items():
            sheet = book[sheet]
            
            # Determine the range for the table
            end_row = sheet.max_row
            end_column = sheet.max_column
            end_column_letter = self.column_number_to_letter(end_column)
            table_range = f"A1:{end_column_letter}{end_row}"
            
            # Create a table
            try:
                table = Table(displayName=table_name, ref=table_range)
                # Add a default style with striped rows
                style = TableStyleInfo(name="TableStyleMedium9", showFirstColumn=False,
                                    showLastColumn=False, showRowStripes=True, showColumnStripes=True)
                table.tableStyleInfo = style
                # Add the table to the sheet
                sheet.add_table(table)
            except ValueError as e:
                print(f"Error creating table for range {table_range}: {e}")
                return
            
            # Adjust column widths
            for col in sheet.columns:
                max_length = 0
                column = col[0].column_letter  # Get the column name
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                sheet.column_dimensions[column].width = adjusted_width

            book.save(file_name)


    def append_rows_to_excel(self, pivot):
        # Convert column names to strings
        pivot.columns = pivot.columns.astype(str)
        
        # Iterate over each row in the pivot DataFrame
        for index, row in pivot.iterrows():
            file_path = os.path.join(self.directory, f"{row['file_name']}.xlsx")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
            with self.setup_excel_writer(file_path, pivot.columns[:-1]) as writer:
                # Convert the single row DataFrame to write to Excel
                row_df = pd.DataFrame([row[:-1]])
                try:
                    book = load_workbook(file_path)
                    sheet = book['Sales']
                    start_row = sheet.max_row
                except Exception:
                    start_row = 2
                    book = writer.book
                    sheet = book.active
                row_df.to_excel(writer, index=False, header=False,
                                startrow=start_row, sheet_name='Sales')
                # Save the workbook
                book.save(file_path)

    def duplicate_sheet_with_zeros(self, file_name):
        # Load the workbook and the sheet to be duplicated
        workbook = load_workbook(file_name)
        original_sheet = workbook['Sales']

        # Duplicate the sheet twice
        for sheet_name in ['Sample', 'Demo']:
            new_sheet = workbook.copy_worksheet(original_sheet)
            new_sheet.title = f"{sheet_name}"

            # Set all cell values to zero
            for row in new_sheet.iter_rows(min_row=2, max_row=new_sheet.max_row,
                                           min_col=5, max_col=new_sheet.max_column):
                for cell in row:
                    cell.value = 0

        # Save the workbook
        workbook.save(file_name)

    def lock_columns(self, file_name, password):
        """Lock the first four columns of all sheets with a password."""
        workbook = load_workbook(file_name)

        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            for row in sheet.iter_rows(min_row=1, max_row=sheet.max_row, min_col=1, max_col=sheet.max_column):
                for cell in row:
                    cell.protection = Protection(locked=False)
            # Lock the first four columns
            for col in range(1, 5):  # Columns A, B, C, D
                for row in sheet.iter_rows(min_col=col, max_col=col, min_row=1, max_row=sheet.max_row):
                    for cell in row:
                        cell.protection = Protection(locked=True)
            sheet.protection.set_password(password)
        workbook.save(file_name)

    def apply_number_format(self, file_name):
        """Apply a number format with a separator from column E to S."""
        workbook = load_workbook(file_name)

        # Create a custom number format style
        number_style = NamedStyle(name="number_style", number_format="#,##0")

        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]

            # Apply number format to columns D to S
            for col in range(5, 20):  # Columns D (4) to S (19)
                for row in sheet.iter_rows(min_col=col, max_col=col, min_row=2, max_row=sheet.max_row):
                    for cell in row:
                        if isinstance(cell.value, (int, float)):
                            cell.style = number_style
        workbook.save(file_name)
                    
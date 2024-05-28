import os
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo

class ExcelManager:
    def __init__(self, directory):
        self.directory = directory

    def setup_excel_writer(self, file_name, df_columns):
        if not os.path.exists(file_name):
            # Create a new workbook and setup headers
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Sheet1"
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
                    self.apply_table_formatting_and_adjust_columns(file_path)

    def column_number_to_letter(self, n):
        """Convert a column number (e.g., 1) to a column letter (e.g., 'A')."""
        string = ""
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            string = chr(65 + remainder) + string
        return string

    def apply_table_formatting_and_adjust_columns(self, file_name):
        book = load_workbook(file_name)
        sheet = book['Sheet1']
        
        # Determine the range for the table
        end_row = sheet.max_row
        end_column = sheet.max_column
        end_column_letter = self.column_number_to_letter(end_column)
        table_range = f"A1:{end_column_letter}{end_row}"
        
        # Create a table
        try:
            table = Table(displayName="Table1", ref=table_range)
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
                    sheet = book['Sheet1']
                    start_row = sheet.max_row
                except Exception:
                    start_row = 2
                    book = writer.book
                    sheet = book.active
                row_df.to_excel(writer, index=False, header=False, startrow=start_row, sheet_name='Sheet1')
                # Save the workbook
                book.save(file_path)
                    
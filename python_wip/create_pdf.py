# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:53:01 2021

@author: Mark Marner-Hausen
"""


# Could create it for multilevel index first and maybe later for single index
# Def a Function, set the first level over multiple sells based on the input 
def create_pdf(Table, MultiIndexLevel, Name_PDF, Header_Size, Header_Lines, Text_Size):
    
    from fpdf import FPDF, fpdf
    import PyPDF2
    #  Define level of index, fontsize 
  
    
    # A4 210 x 297 mm, table width according to regularities (l=35; r=25)
    # Note cell_width depends on the max. amount of columns in the table as well as the alignment (table_width)
    header_height = Header_Size * 0.5
    text_height = Text_Size * 0.4
    table_width = 165
    cell_width = table_width / len(Table.columns)
    
    # Define the Size of the paper used
    paper_width = table_width + 5
    paper_height = (len(Table) * (Text_Size * 0.5)) + (MultiIndexLevel * Header_Size * Header_Lines * 0.5) 
    
    pdf = FPDF('P','mm', (paper_width, paper_height))
    
    # Set the white margins around the text
    pdf.set_auto_page_break(False, 0.5)
    pdf.set_margins(2.5 , 2.5, 1)
    
    
    pdf.add_page()
    
    
    # Initiate space left + Upper line + Font 
    pdf.set_font('Times', 'B', Header_Size)
    pdf.cell(0.1)  
    pdf.cell(table_width, 1, '', 'T', 2, 'C')
    
    # =============================================================================
    # Create the Header
    
    # If we have Multiindexlevel, the header will consist of multiple cells, i.e. must reach further 
    # than 1* cell_width. Yet, the amount of cell_width must not be equal for all header.
    # Further note, if Multiindexlevel = 1 than .columns.levels[XX] does not exist.
    if MultiIndexLevel > 1:
        Table.columns = Table.columns.remove_unused_levels()
        column_names = list(Table.columns.levels[0])
    else:
        column_names = list(Table.columns)
    # top and offset safe the y (stays the same) and x (changes with each loop) position    
    top = pdf.y 
    for column in column_names:
        num_subcol = 1
        if MultiIndexLevel > 1:
            num_subcol = Table.loc[:, column]
            num_subcol = len(num_subcol.columns)
        offset = pdf.x + (cell_width * num_subcol)
        pdf.multi_cell(w = (cell_width * num_subcol), h = header_height, txt = column, border = 'T', align = 'C')
        pdf.y = top
        pdf.x = offset
    
    # Line break     
    pdf.cell(w = 10, h = header_height, txt = '', border = 0, ln = 2, align = 'C')
    if Header_Lines > 1:
        pdf.cell(w = table_width, h = header_height, txt = '', border = 0, ln = 2, align = 'C')
    pdf.cell(w = (-1 * table_width))
    
    # Font for remaining Text
    pdf.set_font('Times', '', Text_Size)
    
    # Second level of Indexes, if existent 
    if MultiIndexLevel > 1:    
        top = pdf.y 
        for column in list(Table.columns.droplevel(0)): 
            offset = pdf.x + cell_width
            pdf.multi_cell(w = cell_width, h = text_height, txt = column, border = 'T', align = 'C')
            pdf.y = top
            pdf.x = offset                          
            
        pdf.cell(w = 10, h = text_height, txt = '', border = 0, ln = 2, align = 'C')
        pdf.cell(w = (-1 * table_width))            

    # =============================================================================
        
    # Create the body of the Table based on the entries position in the matrix, f.ex. (1,1)  
    for row in range(0, len(Table)):
        top = pdf.y 
        for column in range(0, len(Table.columns)):
            offset = pdf.x + cell_width
            pdf.multi_cell(w = cell_width, h = text_height, 
                           txt = str(Table.iloc[row, column]), border = 0, align = 'C')
            pdf.y = top
            pdf.x = offset 

                
        pdf.cell(w = 10, h = text_height, txt = '', border = 0, ln = 2, align = 'C')
        pdf.cell(w = (-1 * table_width))
    
    # Create Bottom line as well as the pdf-Output            
    pdf.cell(table_width, 1, '', 'T', 2, 'C') 
    
    pdf.output(Name_PDF, 'F')   
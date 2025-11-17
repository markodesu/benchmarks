"""
Generate PDF report with all benchmark results
"""

import os
import glob
from tsp_algorithms import parse_tsp_file, run_all_algorithms
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime


def find_benchmark_files(directory: str = ".") -> list:
    """Find all .txt benchmark files"""
    pattern = os.path.join(directory, "*.txt")
    files = glob.glob(pattern)
    return sorted([f for f in files if os.path.isfile(f)])


def run_all_tests():
    """Run all algorithms on all benchmark files"""
    benchmark_files = find_benchmark_files()
    all_results = []
    
    print("Running algorithms on all benchmark files...")
    for filename in benchmark_files:
        try:
            basename = os.path.basename(filename)
            print(f"Processing: {basename}")
            
            graph = parse_tsp_file(filename)
            results = run_all_algorithms(graph, basename)
            all_results.append(results)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    return all_results


def create_pdf(results, output_filename="TSP_Algorithm_Results.pdf"):
    """Create PDF with results table"""
    doc = SimpleDocTemplate(output_filename, pagesize=landscape(letter))
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=1  # Center
    )
    title = Paragraph("TSP Algorithm Benchmark Results", title_style)
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Date
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#666666'),
        alignment=1
    )
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_para = Paragraph(f"Generated on: {date_str}", date_style)
    story.append(date_para)
    story.append(Spacer(1, 0.3*inch))
    
    # Prepare table data
    table_data = []
    
    # Header row
    header = [
        "Benchmark",
        "Cities",
        "Optimal\nDistance",
        "NN Distance",
        "NN Time (s)",
        "GA Distance",
        "GA Time (s)",
        "BF Distance",
        "BF Time (s)",
        "BB Distance",
        "BB Time (s)"
    ]
    table_data.append(header)
    
    # Get optimal distances from Excel
    optimal_distances = {}
    try:
        import openpyxl
        wb = openpyxl.load_workbook('Algorithms(Optimility and TIme Complexity) (1).xlsx')
        ws = wb.active
        for row in ws.iter_rows(min_row=2, max_row=18, values_only=True):
            if row[0] and row[1]:
                name = str(row[0]).strip()
                optimal = row[1]
                optimal_distances[name] = optimal
    except Exception as e:
        print(f"Warning: Could not read optimal distances from Excel: {e}")
    
    # Data rows
    for result in results:
        filename = result['filename'].replace('.txt', '').replace('.tsp', '')
        num_cities = result['num_cities']
        
        # Get optimal distance
        optimal = optimal_distances.get(filename, 'N/A')
        if optimal == 'N/A':
            # Try variations of filename
            for key in optimal_distances:
                if key in filename or filename in key:
                    optimal = optimal_distances[key]
                    break
        
        row = [filename, str(num_cities), str(optimal)]
        
        # Nearest Neighbor
        nn = result['algorithms'].get('NearestNeighbor', {})
        if 'distance' in nn:
            row.append(f"{nn['distance']:.2f}")
            row.append(f"{nn['time']:.4f}")
        else:
            row.append("N/A")
            row.append("N/A")
        
        # Genetic Algorithm
        ga = result['algorithms'].get('GeneticAlgorithm', {})
        if 'distance' in ga:
            row.append(f"{ga['distance']:.2f}")
            row.append(f"{ga['time']:.4f}")
        else:
            row.append("N/A")
            row.append("N/A")
        
        # Brute Force
        bf = result['algorithms'].get('BruteForce', {})
        if 'distance' in bf:
            row.append(f"{bf['distance']:.2f}")
            row.append(f"{bf['time']:.4f}")
        else:
            row.append("N/A")
            row.append("N/A")
        
        # Branch and Bound
        bb = result['algorithms'].get('BranchAndBound', {})
        if 'distance' in bb:
            row.append(f"{bb['distance']:.2f}")
            row.append(f"{bb['time']:.4f}")
        else:
            row.append("N/A")
            row.append("N/A")
        
        table_data.append(row)
    
    # Create table
    table = Table(table_data, colWidths=[1.2*inch, 0.6*inch, 1*inch, 1*inch, 0.8*inch, 
                                         1*inch, 0.8*inch, 1*inch, 0.8*inch, 1*inch, 0.8*inch])
    
    # Style the table
    table.setStyle(TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        
        # Data rows
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F2F2')]),
        
        # First column (benchmark name) - left align
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.3*inch))
    
    # Add note
    note_style = ParagraphStyle(
        'NoteStyle',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#666666'),
        alignment=0  # Left
    )
    note = Paragraph(
        "<b>Note:</b> NN = Nearest Neighbor, GA = Genetic Algorithm, "
        "BF = Brute Force, BB = Branch and Bound. "
        "For instances with >10 cities, Brute Force uses a heuristic. "
        "For instances with >20 cities, Branch and Bound uses Nearest Neighbor heuristic.",
        note_style
    )
    story.append(note)
    
    # Build PDF
    doc.build(story)
    print(f"\nPDF created successfully: {output_filename}")


def main():
    """Main function"""
    print("="*80)
    print("TSP Algorithm Benchmark - PDF Report Generator")
    print("="*80)
    
    # Run all tests
    results = run_all_tests()
    
    if not results:
        print("No results to generate PDF!")
        return
    
    # Create PDF
    print(f"\nGenerating PDF with {len(results)} benchmark results...")
    create_pdf(results)
    
    print("\nDone!")


if __name__ == "__main__":
    main()


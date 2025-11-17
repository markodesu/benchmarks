"""
Generate Comparative Analysis PDF for TSP Algorithms
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
import statistics


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
    
    return all_results


def get_optimal_distances():
    """Get optimal distances from Excel"""
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
    return optimal_distances


def calculate_metrics(results, optimal_distances):
    """Calculate performance metrics"""
    metrics = {
        'nn_ratios': [],
        'ga_ratios': [],
        'bf_ratios': [],
        'bb_ratios': [],
        'nn_times': [],
        'ga_times': [],
        'bf_times': [],
        'bb_times': [],
        'nn_distances': [],
        'ga_distances': [],
        'bf_distances': [],
        'bb_distances': [],
    }
    
    for result in results:
        filename = result['filename'].replace('.txt', '').replace('.tsp', '')
        
        # Get optimal
        optimal = optimal_distances.get(filename, None)
        if optimal is None:
            for key in optimal_distances:
                if key in filename or filename in key:
                    optimal = optimal_distances[key]
                    break
        
        if optimal and isinstance(optimal, (int, float)) and optimal > 0:
            # Nearest Neighbor
            nn = result['algorithms'].get('NearestNeighbor', {})
            if 'distance' in nn:
                ratio = nn['distance'] / optimal
                metrics['nn_ratios'].append(ratio)
                metrics['nn_distances'].append(nn['distance'])
                if 'time' in nn:
                    metrics['nn_times'].append(nn['time'])
            
            # Genetic Algorithm
            ga = result['algorithms'].get('GeneticAlgorithm', {})
            if 'distance' in ga:
                ratio = ga['distance'] / optimal
                metrics['ga_ratios'].append(ratio)
                metrics['ga_distances'].append(ga['distance'])
                if 'time' in ga:
                    metrics['ga_times'].append(ga['time'])
            
            # Brute Force
            bf = result['algorithms'].get('BruteForce', {})
            if 'distance' in bf:
                ratio = bf['distance'] / optimal
                metrics['bf_ratios'].append(ratio)
                metrics['bf_distances'].append(bf['distance'])
                if 'time' in bf:
                    metrics['bf_times'].append(bf['time'])
            
            # Branch and Bound
            bb = result['algorithms'].get('BranchAndBound', {})
            if 'distance' in bb:
                ratio = bb['distance'] / optimal
                metrics['bb_ratios'].append(ratio)
                metrics['bb_distances'].append(bb['distance'])
                if 'time' in bb:
                    metrics['bb_times'].append(bb['time'])
    
    return metrics


def calculate_statistics(values):
    """Calculate statistics for a list of values"""
    if not values:
        return {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0}
    
    return {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'min': min(values),
        'max': max(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0
    }


def create_comparative_pdf(results, output_filename="TSP_Comparative_Analysis.pdf"):
    """Create PDF with comparative analysis"""
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=1
    )
    title = Paragraph("TSP Algorithms - Comparative Analysis", title_style)
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
    story.append(Spacer(1, 0.4*inch))
    
    # Get optimal distances
    optimal_distances = get_optimal_distances()
    
    # Calculate metrics
    metrics = calculate_metrics(results, optimal_distances)
    
    # 1. Performance Ratio Analysis
    story.append(Paragraph("1. Performance Ratio Analysis (Distance / Optimal)", 
                          styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    ratio_data = []
    ratio_data.append(['Algorithm', 'Mean Ratio', 'Median Ratio', 'Min Ratio', 'Max Ratio', 'Std Dev'])
    
    algorithms = [
        ('Nearest Neighbor', metrics['nn_ratios']),
        ('Genetic Algorithm', metrics['ga_ratios']),
        ('Brute Force', metrics['bf_ratios']),
        ('Branch and Bound', metrics['bb_ratios'])
    ]
    
    for algo_name, ratios in algorithms:
        if ratios:
            stats = calculate_statistics(ratios)
            ratio_data.append([
                algo_name,
                f"{stats['mean']:.3f}",
                f"{stats['median']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}",
                f"{stats['std']:.3f}"
            ])
        else:
            ratio_data.append([algo_name, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
    
    ratio_table = Table(ratio_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    ratio_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F2F2')]),
    ]))
    story.append(ratio_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Interpretation
    interpretation = Paragraph(
        "<b>Interpretation:</b> A ratio of 1.0 means the algorithm found the optimal solution. "
        "Ratios < 1.0 indicate better-than-optimal results (may indicate data discrepancy). "
        "Ratios > 1.0 indicate suboptimal solutions. Lower ratios are better.",
        styles['Normal']
    )
    story.append(interpretation)
    story.append(Spacer(1, 0.3*inch))
    
    # 2. Execution Time Analysis
    story.append(Paragraph("2. Execution Time Analysis", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    time_data = []
    time_data.append(['Algorithm', 'Mean Time (s)', 'Median Time (s)', 'Min Time (s)', 'Max Time (s)'])
    
    time_algorithms = [
        ('Nearest Neighbor', metrics['nn_times']),
        ('Genetic Algorithm', metrics['ga_times']),
        ('Brute Force', metrics['bf_times']),
        ('Branch and Bound', metrics['bb_times'])
    ]
    
    for algo_name, times in time_algorithms:
        if times:
            stats = calculate_statistics(times)
            time_data.append([
                algo_name,
                f"{stats['mean']:.4f}",
                f"{stats['median']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}"
            ])
        else:
            time_data.append([algo_name, 'N/A', 'N/A', 'N/A', 'N/A'])
    
    time_table = Table(time_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    time_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#70AD47')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F2F2')]),
    ]))
    story.append(time_table)
    story.append(Spacer(1, 0.3*inch))
    
    # 3. Algorithm Comparison by Benchmark
    story.append(PageBreak())
    story.append(Paragraph("3. Algorithm Comparison by Benchmark", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    comparison_data = []
    comparison_data.append([
        'Benchmark', 'Optimal', 'NN Ratio', 'GA Ratio', 'BF Ratio', 'BB Ratio'
    ])
    
    for result in results:
        filename = result['filename'].replace('.txt', '').replace('.tsp', '')
        
        # Get optimal
        optimal = optimal_distances.get(filename, 'N/A')
        if optimal == 'N/A':
            for key in optimal_distances:
                if key in filename or filename in key:
                    optimal = optimal_distances[key]
                    break
        
        row = [filename, str(optimal)]
        
        # Calculate ratios
        if optimal and isinstance(optimal, (int, float)) and optimal > 0:
            for algo in ['NearestNeighbor', 'GeneticAlgorithm', 'BruteForce', 'BranchAndBound']:
                algo_result = result['algorithms'].get(algo, {})
                if 'distance' in algo_result:
                    ratio = algo_result['distance'] / optimal
                    row.append(f"{ratio:.3f}")
                else:
                    row.append('N/A')
        else:
            row.extend(['N/A', 'N/A', 'N/A', 'N/A'])
        
        comparison_data.append(row)
    
    comparison_table = Table(comparison_data, colWidths=[1.5*inch, 1*inch, 0.9*inch, 
                                                          0.9*inch, 0.9*inch, 0.9*inch])
    comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FFC000')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F2F2')]),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
    ]))
    story.append(comparison_table)
    story.append(Spacer(1, 0.3*inch))
    
    # 4. Key Findings
    story.append(PageBreak())
    story.append(Paragraph("4. Key Findings and Conclusions", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    findings = []
    
    # Find best performing algorithm
    if metrics['nn_ratios']:
        nn_mean = statistics.mean(metrics['nn_ratios'])
        ga_mean = statistics.mean(metrics['ga_ratios']) if metrics['ga_ratios'] else float('inf')
        bf_mean = statistics.mean(metrics['bf_ratios']) if metrics['bf_ratios'] else float('inf')
        bb_mean = statistics.mean(metrics['bb_ratios']) if metrics['bb_ratios'] else float('inf')
        
        algo_means = [
            ('Nearest Neighbor', nn_mean),
            ('Genetic Algorithm', ga_mean),
            ('Brute Force', bf_mean),
            ('Branch and Bound', bb_mean)
        ]
        best_algo = min(algo_means, key=lambda x: x[1] if x[1] != float('inf') else float('inf'))
        findings.append(f"<b>Best Average Performance:</b> {best_algo[0]} with average ratio of {best_algo[1]:.3f}")
    
    # Fastest algorithm
    if metrics['nn_times']:
        nn_time = statistics.mean(metrics['nn_times'])
        ga_time = statistics.mean(metrics['ga_times']) if metrics['ga_times'] else float('inf')
        bf_time = statistics.mean(metrics['bf_times']) if metrics['bf_times'] else float('inf')
        bb_time = statistics.mean(metrics['bb_times']) if metrics['bb_times'] else float('inf')
        
        time_means = [
            ('Nearest Neighbor', nn_time),
            ('Genetic Algorithm', ga_time),
            ('Brute Force', bf_time),
            ('Branch and Bound', bb_time)
        ]
        fastest = min(time_means, key=lambda x: x[1] if x[1] != float('inf') else float('inf'))
        findings.append(f"<b>Fastest Algorithm:</b> {fastest[0]} with average time of {fastest[1]:.4f} seconds")
    
    # Algorithm characteristics
    findings.append("<b>Algorithm Characteristics:</b>")
    findings.append("• <b>Nearest Neighbor:</b> Fast, greedy heuristic. Good for quick solutions but may not be optimal.")
    findings.append("• <b>Genetic Algorithm:</b> Population-based metaheuristic. Can find good solutions but requires more time.")
    findings.append("• <b>Brute Force:</b> Exact algorithm but only feasible for small instances (≤10 cities).")
    findings.append("• <b>Branch and Bound:</b> Exact algorithm with pruning. Efficient for medium-sized instances (≤20 cities).")
    
    # Limitations
    findings.append("<b>Limitations Observed:</b>")
    findings.append("• Some benchmark results show significant discrepancies with optimal values, suggesting possible data format differences.")
    findings.append("• For large instances, exact algorithms (Brute Force, Branch and Bound) use heuristic fallbacks.")
    findings.append("• Genetic Algorithm performance depends on parameter tuning and problem characteristics.")
    
    for finding in findings:
        story.append(Paragraph(finding, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    # Build PDF
    doc.build(story)
    print(f"\nComparative Analysis PDF created successfully: {output_filename}")


def main():
    """Main function"""
    print("="*80)
    print("TSP Algorithm Comparative Analysis - PDF Generator")
    print("="*80)
    
    # Run all tests
    results = run_all_tests()
    
    if not results:
        print("No results to generate PDF!")
        return
    
    # Create PDF
    print(f"\nGenerating Comparative Analysis PDF with {len(results)} benchmark results...")
    create_comparative_pdf(results)
    
    print("\nDone!")


if __name__ == "__main__":
    main()


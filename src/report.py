from fpdf import FPDF
import pandas as pd
import os
from datetime import datetime

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Rapport Morphométrie Xenopus', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(df_results: pd.DataFrame, df_stats: pd.DataFrame, output_path: str):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, f"Date du rapport: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Résumé des Données", ln=True)
    pdf.set_font("Arial", size=11)

    total_images = len(df_results)
    conditions = df_results['Condition'].unique() if not df_results.empty else []

    pdf.cell(0, 8, f"Nombre total d'images analysées: {total_images}", ln=True)
    pdf.cell(0, 8, f"Conditions détectées: {', '.join(conditions)}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Analyse Statistique (vs Témoin)", ln=True)

    if not df_stats.empty:
        pdf.set_font("Arial", 'B', 10)
        col_width = 38
        headers = ["Comparaison", "Med. Temoin", "Med. Cond.", "P-value", "Signif."]

        for h in headers:
            pdf.cell(col_width, 8, h, 1)
        pdf.ln()

        pdf.set_font("Arial", size=10)
        for _, row in df_stats.iterrows():
            pdf.cell(col_width, 8, str(row['Comparaison']), 1)
            pdf.cell(col_width, 8, str(row['Médiane Témoin']), 1)
            pdf.cell(col_width, 8, str(row['Médiane Cond.']), 1)
            pdf.cell(col_width, 8, str(row['P-value (str)']), 1)
            pdf.cell(col_width, 8, str(row['Significativité']), 1)
            pdf.ln()
    else:
        pdf.set_font("Arial", 'I', 11)
        pdf.cell(0, 10, "Aucune statistique significative calculée ou pas de groupe témoin.", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "3. Moyennes par Condition", ln=True)

    if not df_results.empty:
        summary = df_results.groupby('Condition')[['Corps_mm', 'Dist_Yeux_mm', 'Rapport']].mean().reset_index()

        pdf.set_font("Arial", 'B', 10)
        headers = ["Condition", "Corps (mm)", "Yeux (mm)", "Rapport Moyen"]
        col_width = 45

        for h in headers:
            pdf.cell(col_width, 8, h, 1)
        pdf.ln()

        pdf.set_font("Arial", size=10)
        for _, row in summary.iterrows():
            pdf.cell(col_width, 8, str(row['Condition']), 1)
            pdf.cell(col_width, 8, f"{row['Corps_mm']:.3f}", 1)
            pdf.cell(col_width, 8, f"{row['Dist_Yeux_mm']:.3f}", 1)
            pdf.cell(col_width, 8, f"{row['Rapport']:.4f}", 1)
            pdf.ln()

    try:
        pdf.output(output_path)
        return True
    except Exception as e:
        print(f"Erreur PDF: {e}")
        return False

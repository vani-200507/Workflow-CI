WORKFLOW CI MACHINE LEARNING

Project ini menggunakan GitHub Actions untuk menjalankan pipeline CI.

Alur workflow:
1. Code di-push ke GitHub
2. GitHub Actions otomatis berjalan
3. MLProject dijalankan menggunakan MLflow
4. Model training dilakukan otomatis
5. Output disimpan sebagai hasil training

Struktur:
- MLProject: berisi model training dan preprocessing
- .workflow: dokumentasi workflow tugas
- .github/workflows: CI pipeline GitHub Actions

Tools:
- Python
- MLflow
- Scikit-learn
- GitHub Actions
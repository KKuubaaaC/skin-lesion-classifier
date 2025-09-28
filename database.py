import sqlite3
import json
import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

class SkinLesionDB:
    """Database manager for skin lesion analysis application"""
    
    def __init__(self, db_path="data/skin_lesions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Create tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id VARCHAR(50) UNIQUE NOT NULL,
                    age INTEGER,
                    gender VARCHAR(10),
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id VARCHAR(50),
                    image_filename VARCHAR(255),
                    image_hash VARCHAR(64),
                    image_size_kb FLOAT,
                    predicted_class VARCHAR(50),
                    confidence FLOAT,
                    all_predictions TEXT,
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    doctor_notes TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                );
                
                -- Create indexes for better performance
                CREATE INDEX IF NOT EXISTS idx_patient_id ON analyses(patient_id);
                CREATE INDEX IF NOT EXISTS idx_analysis_date ON analyses(analysis_date);
                CREATE INDEX IF NOT EXISTS idx_predicted_class ON analyses(predicted_class);
            ''')
    
    def add_patient(self, patient_id: str, age: Optional[int] = None, 
                   gender: Optional[str] = None, notes: Optional[str] = None) -> bool:
        """Add new patient or update existing one"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO patients (patient_id, age, gender, notes) 
                    VALUES (?, ?, ?, ?)
                ''', (patient_id, age, gender, notes))
                return True
        except Exception as e:
            print(f"Error adding patient: {e}")
            return False
    
    def save_analysis(self, patient_id: str, image_filename: str, 
                     image_data: bytes, predictions: List[Dict], 
                     doctor_notes: Optional[str] = None) -> bool:
        """Save analysis results to database"""
        try:
            # Generate image hash for duplicate detection
            image_hash = hashlib.sha256(image_data).hexdigest()
            image_size_kb = len(image_data) / 1024 if image_data else 0
            
            top_prediction = predictions[0] if predictions else {"class": "unknown", "confidence": 0}
            all_predictions_json = json.dumps(predictions)
            
            with sqlite3.connect(self.db_path) as conn:
                # Check for duplicate analysis
                cursor = conn.execute(
                    "SELECT id FROM analyses WHERE image_hash = ? AND patient_id = ?",
                    (image_hash, patient_id)
                )
                if cursor.fetchone():
                    print("Warning: Duplicate image detected")
                    return False
                
                # Insert new analysis
                conn.execute('''
                    INSERT INTO analyses 
                    (patient_id, image_filename, image_hash, image_size_kb, 
                     predicted_class, confidence, all_predictions, doctor_notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    patient_id, 
                    image_filename,
                    image_hash,
                    round(image_size_kb, 2),
                    top_prediction['class'],
                    round(top_prediction['confidence'], 2),
                    all_predictions_json,
                    doctor_notes
                ))
                return True
        except Exception as e:
            print(f"Error saving analysis: {e}")
            return False
    
    def get_patient_history(self, patient_id: str) -> List[Dict]:
        """Get all analyses for a specific patient"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT a.*, p.age, p.gender 
                    FROM analyses a
                    LEFT JOIN patients p ON a.patient_id = p.patient_id
                    WHERE a.patient_id = ? 
                    ORDER BY a.analysis_date DESC
                ''', (patient_id,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting patient history: {e}")
            return []
    
    def get_all_patients(self) -> List[Dict]:
        """Get list of all patients"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT p.*, 
                           COUNT(a.id) as total_analyses,
                           MAX(a.analysis_date) as last_analysis
                    FROM patients p
                    LEFT JOIN analyses a ON p.patient_id = a.patient_id
                    GROUP BY p.patient_id
                    ORDER BY p.created_at DESC
                ''')
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting patients: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get comprehensive database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Basic counts
                cursor = conn.execute('''
                    SELECT 
                        COUNT(DISTINCT patient_id) as total_patients,
                        COUNT(*) as total_analyses,
                        AVG(confidence) as avg_confidence,
                        MIN(analysis_date) as first_analysis,
                        MAX(analysis_date) as last_analysis
                    FROM analyses
                ''')
                basic_stats = cursor.fetchone()
                
                # Class distribution
                cursor = conn.execute('''
                    SELECT predicted_class, COUNT(*) as count, AVG(confidence) as avg_conf
                    FROM analyses 
                    GROUP BY predicted_class
                    ORDER BY count DESC
                ''')
                class_distribution = cursor.fetchall()
                
                # Recent activity (last 7 days)
                cursor = conn.execute('''
                    SELECT DATE(analysis_date) as date, COUNT(*) as count
                    FROM analyses 
                    WHERE analysis_date >= datetime('now', '-7 days')
                    GROUP BY DATE(analysis_date)
                    ORDER BY date
                ''')
                recent_activity = cursor.fetchall()
                
                return {
                    'basic': basic_stats,
                    'class_distribution': class_distribution,
                    'recent_activity': recent_activity
                }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def search_analyses(self, search_term: str = "", 
                       class_filter: str = "", 
                       date_from: str = "", 
                       date_to: str = "") -> List[Dict]:
        """Search analyses with filters"""
        try:
            query = '''
                SELECT a.*, p.age, p.gender 
                FROM analyses a
                LEFT JOIN patients p ON a.patient_id = p.patient_id
                WHERE 1=1
            '''
            params = []
            
            if search_term:
                query += " AND (a.patient_id LIKE ? OR a.image_filename LIKE ?)"
                params.extend([f"%{search_term}%", f"%{search_term}%"])
            
            if class_filter:
                query += " AND a.predicted_class = ?"
                params.append(class_filter)
            
            if date_from:
                query += " AND DATE(a.analysis_date) >= ?"
                params.append(date_from)
            
            if date_to:
                query += " AND DATE(a.analysis_date) <= ?"
                params.append(date_to)
            
            query += " ORDER BY a.analysis_date DESC LIMIT 100"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error searching analyses: {e}")
            return []
    
    def export_to_csv(self, filename: str = None) -> str:
        """Export all data to CSV"""
        if not filename:
            filename = f"skin_lesion_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Export analyses with patient info
                df = pd.read_sql_query('''
                    SELECT 
                        a.patient_id,
                        p.age,
                        p.gender,
                        a.image_filename,
                        a.predicted_class,
                        a.confidence,
                        a.analysis_date,
                        a.doctor_notes
                    FROM analyses a
                    LEFT JOIN patients p ON a.patient_id = p.patient_id
                    ORDER BY a.analysis_date DESC
                ''', conn)
                
                export_path = Path("exports") / filename
                export_path.parent.mkdir(exist_ok=True)
                df.to_csv(export_path, index=False)
                return str(export_path)
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return ""
    
    def delete_patient(self, patient_id: str) -> bool:
        """Delete patient and all their analyses"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete analyses first (foreign key constraint)
                conn.execute("DELETE FROM analyses WHERE patient_id = ?", (patient_id,))
                # Delete patient
                conn.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
                return True
        except Exception as e:
            print(f"Error deleting patient: {e}")
            return False
import psycopg2
import psycopg2.extras
import json
import hashlib
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

class SkinLesionPostgreSQLDB:
    def __init__(self):
        self.connection_params = {
            'host': 'localhost',
            'port': 5432,
            'database': 'skin_lesion_db',
            'user': 'medical_user',
            'password': 'secure_password_123'
        }
    
    def get_connection(self):
        return psycopg2.connect(**self.connection_params)
    
    def add_patient(self, patient_id: str, age: Optional[int] = None, 
                   gender: Optional[str] = None, notes: Optional[str] = None) -> bool:
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO patients (patient_id, age, gender, notes) 
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (patient_id) 
                DO UPDATE SET age = EXCLUDED.age, gender = EXCLUDED.gender, notes = EXCLUDED.notes
            """, (patient_id, age, gender, notes))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding patient: {e}")
            return False
    
    def save_analysis(self, patient_id: str, image_filename: str, 
                     image_data: bytes, predictions: List[Dict], 
                     doctor_notes: Optional[str] = None) -> bool:
        try:
            image_hash = hashlib.sha256(image_data).hexdigest()
            image_size_kb = len(image_data) / 1024 if image_data else 0
            top_prediction = predictions[0] if predictions else {"class": "unknown", "confidence": 0}
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO analyses 
                (patient_id, image_filename, image_hash, image_size_kb, 
                 predicted_class, confidence, all_predictions, doctor_notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                patient_id, 
                image_filename,
                image_hash,
                round(image_size_kb, 2),
                top_prediction['class'],
                round(top_prediction['confidence'], 2),
                json.dumps(predictions),
                doctor_notes
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving analysis: {e}")
            return False
    
    def get_all_patients(self) -> List[Dict]:
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                SELECT 
                    p.patient_id, p.age, p.gender, p.notes, p.created_at,
                    COUNT(a.id) as total_analyses,
                    MAX(a.analysis_date) as last_analysis
                FROM patients p
                LEFT JOIN analyses a ON p.patient_id = a.patient_id
                GROUP BY p.patient_id, p.age, p.gender, p.notes, p.created_at
                ORDER BY p.created_at DESC
            """)
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return results
        except Exception as e:
            print(f"Error getting patients: {e}")
            return []
    
    def get_patient_history(self, patient_id: str) -> List[Dict]:
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                SELECT a.*, p.age, p.gender 
                FROM analyses a
                LEFT JOIN patients p ON a.patient_id = p.patient_id
                WHERE a.patient_id = %s 
                ORDER BY a.analysis_date DESC
            """, (patient_id,))
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return results
        except Exception as e:
            print(f"Error getting patient history: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Basic stats
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT patient_id) as total_patients,
                    COUNT(*) as total_analyses,
                    ROUND(AVG(confidence), 2) as avg_confidence
                FROM analyses
            """)
            basic_stats = dict(cursor.fetchone()) if cursor.rowcount > 0 else {}
            
            # Class distribution
            cursor.execute("""
                SELECT predicted_class, COUNT(*) as count
                FROM analyses 
                GROUP BY predicted_class
                ORDER BY count DESC
            """)
            class_distribution = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return {
                'basic': basic_stats,
                'class_distribution': class_distribution,
                'recent_activity': []
            }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
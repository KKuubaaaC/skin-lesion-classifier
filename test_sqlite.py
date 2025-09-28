try:
    from database import SkinLesionDB
    print("✅ Import OK")
    
    db = SkinLesionDB()
    print("✅ Database created")
    
    patients = db.get_all_patients()
    print(f"✅ Patients: {len(patients)}")
    
    stats = db.get_statistics()
    print(f"✅ Stats: {stats}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
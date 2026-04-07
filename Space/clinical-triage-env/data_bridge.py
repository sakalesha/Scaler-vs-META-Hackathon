import uuid
import random
from typing import Dict, Any, List
from models import PatientState

class FHIRMapper:
    """
    Maps standard FHIR JSON resources to the ClinicalTriageEnv PatientState.
    Supports Patient, Observation (Vitals), and Condition (Complaints).
    """

    @staticmethod
    def from_fhir_bundle(bundle: Dict[str, Any]) -> PatientState:
        """
        Parses a FHIR Transaction/History bundle and returns a PatientState.
        """
        entries = bundle.get("entry", [])
        patient_data = {}
        vitals = {}
        complaint = "Unknown (Non-FHIR)"

        for entry in entries:
            res = entry.get("resource", {})
            rtype = res.get("resourceType")

            if rtype == "Patient":
                patient_data = {
                    "id": res.get("id", str(uuid.uuid4())[:8]),
                    "gender": "M" if res.get("gender") == "male" else "F",
                    "age_group": FHIRMapper._calculate_age_group(res.get("birthDate")),
                }
            elif rtype == "Condition":
                # Use the clinical display name of the primary condition as the chief complaint
                complaint = res.get("code", {}).get("text", "Clinical presentation")
            elif rtype == "Observation":
                # Map LOINC codes to our vital sign fields
                FHIRMapper._map_observation(res, vitals)

        # Build the model
        return PatientState(
            patient_id=patient_data.get("id", str(uuid.uuid4())[:8]),
            wait_minutes=0.0,
            chief_complaint=complaint,
            age_group=patient_data.get("age_group", "adult"),
            gender=patient_data.get("gender", "F"),
            true_esi=FHIRMapper._infer_esi_from_vitals(vitals),
            **vitals
        )

    @staticmethod
    def _calculate_age_group(birth_date: str) -> str:
        if not birth_date: return "adult"
        # Simplified logic for hackathon
        year = int(birth_date.split("-")[0])
        age = 2026 - year
        if age < 2: return "infant"
        if age < 18: return "child"
        if age > 65: return "elderly"
        return "adult"

    @staticmethod
    def _map_observation(res: Dict[str, Any], vitals: Dict[str, Any]):
        loinc = res.get("code", {}).get("coding", [{}])[0].get("code")
        val = res.get("valueQuantity", {}).get("value")

        if loinc == "8867-4": # Heart Rate
            vitals["heart_rate"] = int(val)
        elif loinc == "8480-6": # Systolic BP
            vitals["systolic_bp"] = int(val)
        elif loinc == "2708-6": # SpO2
            vitals["spo2"] = float(val) / 100.0 if val > 1 else float(val)
        elif loinc == "9279-1": # Respiratory Rate
            vitals["respiratory_rate"] = int(val)
        elif loinc == "8310-5": # Temperature
            vitals["temperature"] = float(val)

    @staticmethod
    def _infer_esi_from_vitals(vitals: Dict[str, Any]) -> int:
        """Heuristic ESI inference for real-time data playback."""
        hr = vitals.get("heart_rate", 80)
        spo2 = vitals.get("spo2", 0.98)
        if hr > 130 or spo2 < 0.90: return 1
        if hr > 110 or spo2 < 0.93: return 2
        if hr > 100: return 3
        return 4

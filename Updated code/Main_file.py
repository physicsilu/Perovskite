import pandas as pd
import numpy as np
import joblib
import os
from pymatgen.core import Element
from itertools import product

# Input and target columns for the model
input_cols = [
    'mean_at_num', 'std_at_num',
    'mean_grp_num', 'std_grp_num',
    'mean_mend_num', 'std_mend_num',
    'mean_row_num', 'std_row_num',
    'mean_atomic_radius', 'std_atomic_radius',
    'mean_ionic_radius', 'std_ionic_radius',
    'mean_van_der_waal_radius', 'std_van_der_waal_radius',
    'mean_atomic_radius_calculated', 'std_atomic_radius_calculated',
    'mean_covalent_radius', 'std_covalent_radius',
    'mean_electron_affinity', 'std_electron_affinity',
    'mean_electronegativity', 'std_electronegativity',
    'mean_first_ionization_energy', 'std_first_ionization_energy',
    'mean_dipole_polarizability', 'std_dipole_polarizability',
    'mean_atomic_mass', 'std_atomic_mass',
    'mean_density', 'std_density',
    'mean_molar_volume', 'std_molar_volume',
    'mean_boiling_point', 'std_boiling_point',
    'mean_melting_point', 'std_melting_point',
    'mean_thermal_conductivity', 'std_thermal_conductivity',
    'mean_specific_heat', 'std_specific_heat'
]

target_col = 'band_gap'

def get_element_properties(element_symbol):
    """Get elemental properties."""
    try:
        element = Element(element_symbol)
        return {
            'atomic_number': element.Z,
            'group': element.group,
            'row': element.row,
            'electronegativity': element.X,
            'atomic_radius': element.atomic_radius,
            'atomic_radius_calculated': element.atomic_radius_calculated,
            'van_der_waals_radius': element.van_der_waals_radius,
            'electron_affinity': element.electron_affinity,
            'ionization_energy': element.ionization_energies[0] if element.ionization_energies else np.nan
        }
    except:
        return {k: np.nan for k in ['atomic_number', 'group', 'row', 'electronegativity',
                                    'atomic_radius', 'atomic_radius_calculated',
                                    'van_der_waals_radius', 'electron_affinity',
                                    'ionization_energy']}


# add more inputs


def prepare_input_features(elements, model_features):
    """Prepare input features matching model requirements."""
    properties = [get_element_properties(elem) for elem in elements]
    
    features = {}
    for prop in properties[0].keys():
        values = [p[prop] for p in properties]
        features[f'mean_{prop}'] = np.mean(values)
        features[f'std_{prop}'] = np.std(values)
    
    df = pd.DataFrame([features])
    df = df.reindex(columns=model_features, fill_value=np.nan)
    return df

def load_model(model_name):
    """Load the pre-trained model."""
    model_paths = {
        'LightGBM': "LightGBM_model.pkl",
        'XGBoost': "XGBoost_model.pkl",
        'Random Forest': "Random Forest_model.pkl",
        'Gradient Boosting': "Gradient Boosting_model.pkl"
    }
    
    if model_name not in model_paths or not os.path.exists(model_paths[model_name]):
        raise FileNotFoundError(f"Model file for '{model_name}' not found. Available models are: {list(model_paths.keys())}")
    
    model = joblib.load(model_paths[model_name])
    return model

def predict_bandgap(elements, model):
    """Predict bandgap with correct features."""
    input_features = prepare_input_features(elements, model.feature_names_in_)
    return model.predict(input_features)[0]

def generate_materials(model_name, bandgap_min, bandgap_max):
    """Generate potential perovskite compositions and filter based on predicted bandgap."""
    possible_A = ["Cs", "Ba", "Sr", "Ca", "K", "Na", "La"] # Group 1 and Group 2
    possible_B = ["Nb", "Ta", "Ti", "Zr", "Hf", "Mo", "Al"] # transition matel
    possible_O = ["O"]
    
    compositions = list(product(possible_A, possible_B, possible_O))
    results = []
    
    model = load_model(model_name)
    for A, B, O in compositions:
        elements = [A, B, O]
        try:
            predicted_bandgap = predict_bandgap(elements, model)
            if bandgap_min <= predicted_bandgap <= bandgap_max:
                results.append((f"{A}{B}{O}3", predicted_bandgap))
        except:
            continue
    
    if not results:
        print(f"\nNo perovskite compositions found with bandgap between {bandgap_min:.2f} and {bandgap_max:.2f} eV")
    else:
        results.sort(key=lambda x: x[1])
        print(f"\nPredicted Perovskite Materials (Bandgap {bandgap_min:.2f} - {bandgap_max:.2f} eV):")
        print("=====================================")
        for formula, bandgap in results:
            print(f"{formula:<15} {bandgap:.3f} eV")
        print(f"\nTotal materials found: {len(results)}")

def main():
    print("\nAvailable models:")
    models = ['LightGBM', 'XGBoost', 'Random Forest', 'Gradient Boosting']
    for i, model_name in enumerate(models, 1):
        print(f"{i}. {model_name}")
    
    while True:
        try:
            choice = input("\nEnter the number of the model you want to use (1-4): ")
            model_name = models[int(choice)-1]
            break
        except (ValueError, IndexError):
            print("Invalid choice. Please enter a number between 1 and 4.")
    
    while True:
        try:
            bandgap_min = float(input("Minimum bandgap: "))
            bandgap_max = float(input("Maximum bandgap: "))
            if bandgap_min < 0 or bandgap_max < 0 or bandgap_min >= bandgap_max:
                print("Error: Invalid bandgap range.")
                continue
            generate_materials(model_name, bandgap_min, bandgap_max)
            break
        except ValueError:
            print("Error: Please enter valid numerical values for the bandgap range.")
            continue

if __name__ == "__main__":
    main()

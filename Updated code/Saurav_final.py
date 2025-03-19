import pandas as pd
import numpy as np
import joblib
import os
from pymatgen.core import Element as pym_element
from mendeleev import element as mend_element
from itertools import product

# Define possible elements
group_1_2 = ["Cs", "Ba", "Sr", "Ca", "K", "Na", "La"]  # Group 1 and 2
transition_metals = ["Nb", "Ta", "Ti", "Zr", "Hf", "Mo", "Al"]  # Transition metals
oxygen = ["O"]

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

# Create dictionary to store element properties
def extract_element_properties(elements):
    """
    Extract and store elemental properties from pymatgen and mendeleev.
    """
    element_data = {}
    
    mend_properties = ['covalent_radius', 'fusion_heat', 'atomic_weight', 'specific_heat',
                       'evaporation_heat', 'dipole_polarizability', 'density']
    
    pym_properties = ['group', 'row', 'electronegativity', 'atomic_radius', 'atomic_radius_calculated',
                      'van_der_waals_radius', 'mendeleev_no', 'molar_volume',
                      'electron_affinity', 'ionization_energies', 'average_ionic_radius',
                      'density_of_solid', 'boiling_point', 'melting_point', 'thermal_conductivity']
    
    for elem in elements:
        properties = {}
        
        try:
            mend_elem = mend_element(elem)
            for prop in mend_properties:
                properties[prop] = getattr(mend_elem, prop, np.nan)
        except Exception as e:
            properties.update({prop: np.nan for prop in mend_properties})
            print(f"Warning: Could not fetch mendeleev data for {elem} - {e}")
        
        try:
            pym_elem_obj = pym_element(elem)
            for prop in pym_properties:
                if prop == "ionization_energies":
                    properties["ionization_energy"] = pym_elem_obj.ionization_energies[0] if pym_elem_obj.ionization_energies else np.nan
                else:
                    properties[prop] = getattr(pym_elem_obj, prop, np.nan)
        except Exception as e:
            properties.update({prop: np.nan for prop in pym_properties})
            print(f"Warning: Could not fetch pymatgen data for {elem} - {e}")
        
        element_data[elem] = properties
    
    return element_data

# Fetch and store properties for all possible elements
unique_elements = group_1_2 + transition_metals + oxygen
element_properties = extract_element_properties(unique_elements)

# Function to calculate weighted mean
def compute_weighted_mean(properties, elements):
    weights = [1/len(elements)] * len(elements)
    mean_values = {}
    std_values = {}
    
    for prop in properties[0].keys():
        values = [p[prop] for p in properties]
        values = np.array(values, dtype=np.float64)
        valid_values = values[~np.isnan(values)]
        
        if valid_values.size > 0:
            mean_values[f'mean_{prop}'] = np.average(valid_values, weights=weights[:len(valid_values)])
            std_values[f'std_{prop}'] = np.std(valid_values)
        else:
            mean_values[f'mean_{prop}'] = np.nan
            std_values[f'std_{prop}'] = np.nan
    
    return {**mean_values, **std_values}

# Prepare input features for the model
def prepare_input_features(elements, model_features):
    """Prepare weighted mean and std deviation of elemental properties."""
    properties = [element_properties[elem] for elem in elements]
    features = compute_weighted_mean(properties, elements)
    
    df = pd.DataFrame([features])
    df = df.reindex(columns=model_features, fill_value=np.nan)
    return df

# Load pre-trained model
def load_model(model_name):
    model_paths = {
        'LightGBM': "LightGBM_model.pkl",
        'XGBoost': "XGBoost_model.pkl",
        'Random Forest': "Random Forest_model.pkl",
        'Gradient Boosting': "Gradient Boosting_model.pkl"
    }
    
    if model_name not in model_paths or not os.path.exists(model_paths[model_name]):
        raise FileNotFoundError(f"Model file for '{model_name}' not found.")
    
    return joblib.load(model_paths[model_name])

# Predict bandgap
def predict_bandgap(elements, model):
    input_features = prepare_input_features(elements, model.feature_names_in_)
    return model.predict(input_features)[0]

# Generate perovskite materials
def generate_materials(model_name, bandgap_min, bandgap_max):
    compositions = list(product(group_1_2, transition_metals, oxygen))
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
        print(f"No perovskite compositions found with bandgap between {bandgap_min:.2f} and {bandgap_max:.2f} eV")
    else:
        results.sort(key=lambda x: x[1])
        print("Predicted Perovskite Materials:")
        for formula, bandgap in results:
            print(f"{formula:<15} {bandgap:.3f} eV")
        print(f"Total materials found: {len(results)}")

# Main function to run the prediction
def main():
    models = ['LightGBM', 'XGBoost', 'Random Forest', 'Gradient Boosting']
    for i, model_name in enumerate(models, 1):
        print(f"{i}. {model_name}")
    
    while True:
        try:
            choice = int(input("Enter model number (1-4): "))
            model_name = models[choice - 1]
            break
        except (ValueError, IndexError):
            print("Invalid choice. Try again.")
    
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
            print("Error: Please enter valid numerical values.")

if __name__ == "__main__":
    main()

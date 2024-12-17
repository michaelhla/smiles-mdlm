from flask import Flask, render_template_string, request, jsonify
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import tempfile
import os

app = Flask(__name__)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Molecular Structure Viewer</title>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #viewer { height: 500px; width: 100%; }
        .controls { margin-bottom: 20px; }
        select, input, button { margin: 5px; padding: 5px; }
    </style>
</head>
<body>
    <div class="controls">
        <select id="moleculeSelect">
            <option value="Aspirin">Aspirin</option>
            <option value="Caffeine">Caffeine</option>
            <option value="Ibuprofen">Ibuprofen</option>
            <option value="Water">Water</option>
            <option value="Ethanol">Ethanol</option>
        </select>
        <input type="text" id="smilesInput" placeholder="Or enter SMILES string...">
        <button onclick="generateStructure()">Generate</button>
    </div>
    <div id="viewer"></div>
    <div id="status"></div>

    <script>
        let viewer = null;
        
        function initViewer() {
            viewer = $3Dmol.createViewer("viewer");
            generateStructure();
        }

        function generateStructure() {
            const smiles = document.getElementById('smilesInput').value;
            const selected = document.getElementById('moleculeSelect').value;
            
            fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    smiles: smiles,
                    selected: selected
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('status').textContent = data.error;
                    return;
                }
                
                viewer.clear();
                viewer.addModel(data.pdb, "pdb");
                viewer.setStyle({}, {stick:{}});
                viewer.zoomTo();
                viewer.render();
                document.getElementById('status').textContent = data.status;
            })
            .catch(error => {
                document.getElementById('status').textContent = 'Error: ' + error;
            });
        }

        window.onload = initViewer;
    </script>
</body>
</html>
"""

# Dictionary of example molecules
EXAMPLE_MOLECULES = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)[C@H](C)C(=O)O",
    "Water": "O",
    "Ethanol": "CCO"
}

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        smiles = data.get('smiles', '')
        selected = data.get('selected', '')

        # Get SMILES either from input or selection
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return jsonify({'error': 'Invalid SMILES string'})
        else:
            smiles = EXAMPLE_MOLECULES[selected]
            mol = Chem.MolFromSmiles(smiles)

        # Generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        # Save to temporary PDB file
        temp_dir = tempfile.gettempdir()
        temp_pdb = os.path.join(temp_dir, "temp_molecule.pdb")
        writer = Chem.PDBWriter(temp_pdb)
        writer.write(mol)
        writer.close()

        # Read PDB content
        with open(temp_pdb, 'r') as f:
            pdb_content = f.read()

        # Clean up
        os.remove(temp_pdb)

        return jsonify({
            'pdb': pdb_content,
            'status': f'Showing structure for {smiles}'
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Note: In production, you should use a proper WSGI server
    app.run(host='0.0.0.0', port=5000, debug=True)
import numpy as np
import argparse
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='finename of sdf')
    parser.add_argument('--fptype', type=str, help='ECFP, FCFP, ', default='ECFP')
    parser.add_argument('--radius', type=int, help='radius of ECFP, FCFP ', default=2)
    parser.add_argument('--nBits', type=int, help='number of bits ', default=1024)
    parser.add_argument('--molid', type=str, help='molid prop', default=None)
    parser.add_argument('--target', type=str, help='target name for predict', default=None)
    parser.add_argument('--output', type=str, help='output path', default='data')
    return parser

def fp2list(fp):
    arr = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def mol2fp(mol, fptype='ECFP', radius=2, nBits=1024):
    if fptype=='ECFP':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    elif fptype=='FCFP':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useFeatures=True)
    else:
        raise NotImplementedError('This FP is not implemented!!!! pls use ECFP or FCFP')
    return fp

if __name__=='__main__':
    parser = create_parser()
    args = parser.parse_args()
    mols = Chem.SDMolSupplier(args.input)
    fps = []
    targets = []
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    w = open(os.path.join(args.output,'log.txt'), 'w')
    w.write('mol_id,smiles\n')
    for i, mol in enumerate(mols):
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            if args.molid is not None:
                mol_id =mol.GetProp(args.molid)
            else:
                idx = 1+i
                mol_id = f'mol_{idx}'
            fp = mol2fp(mol,args.fptype, args.radius, args.nBits)
            fp_list = fp2list(fp)
            fps.append(fp_list)
            w.write(f'{mol_id},{smiles}\n')
            if args.target is not None:
                target = mol.GetProp(args.target)
                targets.append(np.float(target))
    w.close()
    X = np.asarray(fps)
    X = np.savez(f'{args.output}/{args.fptype}_{args.radius}_{args.nBits}_X.npz', X)
    if args.target is not None:
        Y = np.asarray(targets)
        np.savez(f'{args.output}/{args.target}_arr.npz', Y)
    else:
        pass

#!/usr/bin/env python

"""
Classes and functions that help reduce molecular simulations to time series data.
In the future, this functionality will get a separate module (this is taken from LLC_Membranes at the moment)
"""

import os
import numpy as np
import matplotlib.path as mplPath
from hdphmm.utils import file_rw

ions_mw = dict()
ions_mw['NA'] = 22.99
ions_mw['BR'] = 79.904

script_location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class ReadItp(object):
    """ Read and store information from a GROMACS topology file

    :param name: name of GROMACS topology file

    :return: charge,
    """

    def __init__(self, name):
        """ Read in .itp file and initialize data structures

        :param name: name of itp (without extension)

        :type name: str
        """

        try:
            f = open('%s.itp' % name, 'r')
        except FileNotFoundError:
            try:
                f = open('%s/../data/%s.itp' % (script_location, name), 'r')
            except FileNotFoundError:
                raise FileNotFoundError('No topology %s.itp found' % name)

        self.itp = []
        for line in f:
            self.itp.append(line)

        f.close()

        # Intialize atom properties
        self.natoms = 0
        self.indices = {}  # key = atom name , value = index
        self.names = {}  # key = index, value = atom name
        self.mass = {}  # key = atom name, value = mass
        self.charges = {}
        self.atom_info = []  # stores all fields of [ atoms ] section of itp file

        # Initialize annotation lists
        self.hbond_H = []  # hydrogen atoms capable of hbonding
        self.hbond_D = []  # hydrogen bond donor atoms
        self.hbond_A = []  # hydrogen bond acceptors
        self.residues = []
        self.planeatoms = []  # names of atoms defining plane used to align monmers
        self.plane_indices = []  # indices of atoms in planeatoms
        self.benzene_carbons = []  # names of atoms making up aromatic ring
        self.lineatoms = [[], []]  # indices of atoms used to create a vector used to orient monomers during build
        self.ref_atom_index = []  # index of atom(s) used as a reference for translating a molecule around during build
        self.c1_atoms = []  # terminal carbon atoms of tails (for cross-linking)
        self.c2_atoms = []  # 2nd carbon atoms from end of monomers tails (for cross-linking)
        self.c1_index = []  # indices of terminal carbon atoms of tails
        self.c2_index = []  # indices of c2_atoms
        self.ion_indices = []  # indices of ions
        self.tail_atoms = []  # names of atoms at ends of tails. Used for placing solutes near tail ends
        self.no_ions = 0  # number of ions in system
        self.ions = []  # names of ions
        self.MW = 0  # molecular weight of system
        self.dummies = []  # names of dummy atoms
        self.valence = 0
        self.carboxylate_indices = []  # index of carboxylate carbon atoms on head groups
        self.pore_defining_atoms = []  # atoms used to define the edges of the pore. used to locate pore center
        self.build_restraints = [] # atoms that will be restrained during build procedure

        # atoms that are a part of improper dihedrals which should not be removed during cross-linking. For example, in
        # NA-GA3C11, the tail has connectivity R-C=O-CH-CH2. When the C2 bonds, it becomes sp3 hybridized and its
        # improper dihedral must be removed. But we don't want to accidentally remove the carbonyl's improper which
        # still involves c2. Specifying the oxygen atom index in improper_dihedral_exclusions prevents this.
        self.improper_dihedral_exclusions = []

        # connectivity
        self.bonds = []  # stores all data in [ bonds ] section of topology
        self.organized_bonds = {}  # a dictionary of atom indices with values that are indices to which atom is bonded
        self.improper_dihedrals = []  # stores all data in [ dihedrals ] ; impropers section of topology
        self.virtual_sites = []  # stores all data in [ virtualsites* ] of topology

    def atoms(self, annotations=False):
        """ Read itp line-by-line, extract annotations (optional) and determine number of atoms, atom indices, names,
        mass, charges and residue molecular weight

        :param annotations: If True, read annotations

        :type annotations: bool
        """

        atoms_index = 0
        while self.itp[atoms_index].count('[ atoms ]') == 0:
            atoms_index += 1

        atoms_index += 2

        while self.itp[self.natoms + atoms_index] != '\n':

            data = self.itp[self.natoms + atoms_index].split()

            self.atom_info.append(data)

            _, type, _, resname, atom_name, _, _, _ = data[:8]

            ndx = int(data[0]) - 1
            charge = float(data[6])

            try:
                mass = float(data[7])
            except ValueError:  # Throws error if annotation is present with semi colon directly next to mass
                mass = float(data[7].split(';')[0])

            self.indices[atom_name] = ndx
            self.names[ndx] = atom_name
            self.mass[atom_name] = float(mass)
            self.charges[atom_name] = float(charge)
            self.MW += mass

            if resname not in self.residues:
                self.residues.append(resname)

            if annotations:

                try:  # check for annotations on atom

                    annotations = self.itp[self.natoms + atoms_index].split(';')[1].split()

                    if 'H' in annotations:
                        self.hbond_H.append(atom_name)
                    if 'D' in annotations:
                        self.hbond_D.append(atom_name)
                    if 'A' in annotations:
                        self.hbond_A.append(atom_name)
                    if 'P' in annotations:
                        self.planeatoms.append(atom_name)
                        self.plane_indices.append(ndx)
                    if 'L1' in annotations:
                        self.lineatoms[1].append(ndx)
                    if 'L2' in annotations:
                        self.lineatoms[0].append(ndx)
                    if 'R' in annotations:
                        self.ref_atom_index.append(ndx)
                    if 'C1' in annotations:
                        self.c1_atoms.append(atom_name)
                        self.c1_index.append(ndx)
                    if 'C2' in annotations:
                        self.c2_atoms.append(atom_name)
                        self.c2_index.append(ndx)
                    if 'I' in annotations:
                        self.no_ions += 1
                        self.valence = charge
                        if atom_name not in self.ions:
                            self.ions.append(atom_name)
                        self.ion_indices.append(ndx)
                    if 'B' in annotations:
                        self.benzene_carbons.append(atom_name)
                    if 'C' in annotations:
                        self.carboxylate_indices.append(ndx)
                    if 'PDA' in annotations:
                        self.pore_defining_atoms.append(atom_name)
                    if 'T' in annotations:
                        self.tail_atoms.append(atom_name)
                    if 'D' in annotations:
                        self.dummies.append(atom_name)
                    if 'impex' in annotations:
                        self.improper_dihedral_exclusions.append(ndx)
                    if 'Rb' in annotations:
                        self.build_restraints.append(atom_name)

                except IndexError:
                    pass

            self.natoms += 1

    def organize_bonds(self):
        """ Determine how each atom is bonded

        :return: A dict with keys that are atom indices and values that are all of the atom indices to which they are
        bonded
        :rtype: dict
        """

        # find the bonds section
        bonds_index = 0
        while self.itp[bonds_index].count('[ bonds ]') == 0:
            bonds_index += 1
        bonds_index += 2

        bonds = []
        while self.itp[bonds_index] != '\n':
            bond_data = str.split(self.itp[bonds_index])[:2]
            bonds.append([int(bond_data[0]), int(bond_data[1])])
            bonds_index += 1

        for i in range(self.natoms):
            self.organized_bonds[i] = []
            involvement = [x for x in bonds if i + 1 in x]
            for pair in involvement:
                atom = [x - 1 for x in pair if x != (i + 1)][0]
                self.organized_bonds[i].append(atom)

    def get_bonds(self):
        """ Store all information in the "[ bonds ]" section of name.itp
        """

        bonds_index = 0
        while self.itp[bonds_index].count('[ bonds ]') == 0:
            bonds_index += 1
        bonds_index += 1

        while self.itp[bonds_index].split()[0] == ';':
            bonds_index += 1

        while self.itp[bonds_index] != '\n':
            self.bonds.append([int(self.itp[bonds_index].split()[i]) for i in range(2)])
            bonds_index += 1

    def get_improper_dihedrals(self):
        """ Store all information in the "[ dihedrals ] ; impropers" section of name.itp
        """

        imp_ndx = 0

        while self.itp[imp_ndx].count('[ dihedrals ] ; impropers') == 0:
            imp_ndx += 1
            if imp_ndx >= len(self.itp):
                break

        if imp_ndx < len(self.itp):
            imp_ndx += 1
            while self.itp[imp_ndx][0] == ';':
                imp_ndx += 1
            while imp_ndx < len(self.itp) and self.itp[imp_ndx] != '\n':
                self.improper_dihedrals.append(self.itp[imp_ndx].split())
                imp_ndx += 1
        else:
            self.improper_dihedrals = None

    def get_vsites(self):
        """ Store all information in the "[ virtual_sites ]" section of name.itp
        """

        vsite_index = 0
        while self.itp[vsite_index].count('[ virtual_sites4 ]') == 0:
            vsite_index += 1
            if vsite_index >= len(self.itp):
                break

        if vsite_index < len(self.itp):
            vsite_index += 1
            while self.itp[vsite_index][0] == ';':
                vsite_index += 1
            while vsite_index < len(self.itp):
                self.virtual_sites.append(self.itp[vsite_index].split())
                vsite_index += 1
        else:
            self.virtual_sites = None


class Residue(ReadItp):

    def __init__(self, name, connectivity=False):
        """ Get attributes of residue based on an .itp file

        :param name: name of .itp file (no extension)
        :param connectivity: get bonds, improper dihedrals and virtual sites

        :type name: str
        :type connectivity: bool
        """

        self.name = name

        self.is_ion = False
        # check if residue is an ion
        with open('%s/../data/ions.txt' % script_location) as f:
            ions = []
            for line in f:
                if line[0] != '#':
                    ions.append(str.strip(line))

        if name in ions:

            self.is_ion = True
            self.natoms = 1
            self.MW = ions_mw[name]
            self.mass = dict()  # key = atom name, value = mass
            self.mass[name] = ions_mw[name]

        else:

            super().__init__(name)

            self.atoms(annotations=True)  # read annotations
            self.organize_bonds()  # might make a flag for this

            if connectivity:
                self.get_bonds()
                self.get_improper_dihedrals()
                self.get_vsites()


def put_in_box(pt, x_box, y_box, m, angle):
    """
    :param pt: The point to place back in the box
    :param x_box: length of box in x dimension
    :param y_box: length of box in y dimension
    :param m: slope of box vector
    :param angle: angle between x axis and y box vector
    :return: coordinate shifted into box
    """

    b = - m * x_box  # y intercept of box vector that does not pass through origin (right side of box)

    if pt[1] < 0:
        pt[:2] += [np.cos(angle)*x_box, np.sin(angle)*x_box]  # if the point is under the box
    if pt[1] > y_box:
        pt[:2] -= [np.cos(angle)*x_box, np.sin(angle)*x_box]
    if pt[1] > m*pt[0]:  # if the point is on the left side of the box
        pt[0] += x_box
    if pt[1] < (m*pt[0] + b):  # if the point is on the right side of the box
        pt[0] -= x_box

    return pt


def trace_pores(pos, box, npoints, npores=4, progress=True, save=True, savename='spline.pl'):
    """
    Find the line which traces through the center of the pores
    :param pos: positions of atoms used to define pore location (args.ref) [natoms, 3]
    :param box: xy box vectors, [2, 2], mdtraj format (t.unitcell_vectors)
    :param npoints: number of points for spline in each pore
    :param npores: number of pores in unit cell (assumed that atoms are number sequentially by pore. i.e. pore 1 atom
    numbers all precede those in pore 2)
    :param progress: set to True if you want a progress bar to be shown
    :param save: save spline as pickled object
    :param savename: path to spline. If absolute path is not provided, will look in current directory

    :type pos: np.ndarray
    :type box: np.ndarray
    :type npoints: int
    :type npores: int
    :type progress: bool
    :type save: bool
    :type savename: str

    :return: points which trace the pore center
    """

    try:
        print('Attempting to load spline ... ', end='', flush=True)
        spline = file_rw.load_object(savename)
        print('Success!')

        return spline[0], spline[1]

    except FileNotFoundError:

        print('%s not found ... Calculating spline' % savename)

        single_frame = False
        if np.shape(pos.shape)[0] == 2:
            pos = pos[np.newaxis, ...]  # add a new axis if we are looking at a single frame
            box = box[np.newaxis, ...]
            single_frame = True

        nframes = pos.shape[0]
        atoms_p_pore = int(pos.shape[1] / npores)  # atoms in each pore

        v = np.zeros([nframes, 4, 2])  # vertices of unitcell box
        bounds = []

        v[:, 0, :] = [0, 0]
        v[:, 1, 0] = box[:, 0, 0]
        v[:, 3, :] = np.vstack((box[:, 1, 0], box[:, 1, 1])).T
        v[:, 2, :] = v[:, 3, :] + np.vstack((box[:, 0, 0], np.zeros([nframes]))).T
        center = np.vstack((np.mean(v[..., 0], axis=1), np.mean(v[..., 1], axis=1), np.zeros(nframes))).T

        for t in range(nframes):
            bounds.append(mplPath.Path(v[t, ...]))  # create a path tracing the vertices, v

        angle = np.arcsin(box[:, 1, 1]/box[:, 0, 0])  # specific to case where magnitude of x and y box lengths are equal
        angle = np.where(box[:, 1, 0] < 0, angle + np.pi / 2, angle)  # haven't tested this well yet

        m = (v[:, 3, 1] - v[:, 0, 1]) / (v[:, 3, 0] - v[:, 0, 0])  # slope from points connecting first and third vertices

        centers = np.zeros([nframes, npores, npoints, 3])
        bin_centers = np.zeros([nframes, npores, npoints])

        for t in tqdm.tqdm(range(nframes), disable=(not progress)):
            for p in range(npores):

                pore = pos[t, p*atoms_p_pore:(p+1)*atoms_p_pore, :]  # coordinates for atoms belonging to a single pore

                while np.min(pore[:, 2]) < 0 or np.max(pore[:, 2]) > box[t, 2, 2]:  # because cross-linked configurations can extend very far up and down

                    pore[:, 2] = np.where(pore[:, 2] < 0, pore[:, 2] + box[t, 2, 2], pore[:, 2])
                    pore[:, 2] = np.where(pore[:, 2] > box[t, 2, 2], pore[:, 2] - box[t, 2, 2], pore[:, 2])

                _, bins = np.histogram(pore[:, 2], bins=npoints)  # bin z-positions

                section_indices = np.digitize(pore[:, 2], bins)  # list that tells which bin each atom belongs to
                bin_centers[t, p, :] = [(bins[i] + bins[i + 1])/2 for i in range(npoints)]

                for l in range(1, npoints + 1):

                    atom_indices = np.where(section_indices == l)[0]

                    before = pore[atom_indices[0], :]  # choose the first atom as a reference

                    shift = translate(pore[atom_indices, :], before, center[t, :])  # shift everything to towards the center

                    for i in range(shift.shape[0]):  # check if the points are within the bounds of the unitcell
                        while not bounds[t].contains_point(shift[i, :2]):
                            shift[i, :] = put_in_box(shift[i, :], box[t, 0, 0], box[t, 1, 1], m[t], angle[t])  # if its not in the unitcell, shift it so it is

                    c = [np.mean(shift, axis=0)]

                    centers[t, p, l - 1, :] = translate(c, center[t, :], before)  # move everything back to where it was

                    while not bounds[t].contains_point(centers[t, p, l - 1, :]):  # make sure everything is in the box again
                        centers[t, p, l - 1, :] = put_in_box(centers[t, p, l - 1, :], box[t, 0, 0], box[t, 1, 1], m[t], angle[t])

        if single_frame:
            return centers[0, ...]  # doesn't return bin center yet

        else:

            if save:
                file_rw.save_object((centers, bin_centers), savename)

            return centers, bin_centers


def center_of_mass(pos, mass_atoms):
    """ Calculate center of mass of residues over a trajectory

    :param pos: xyz coordinates of atoms
    :param mass_atoms : mass of atoms in order they appear in pos

    :type pos: np.array (nframes, natoms, 3)
    :type mass_atoms: list

    :return: center of mass of each residue at each frame
    """

    nframes = pos.shape[0]
    natoms = len(mass_atoms)

    com = np.zeros([nframes, pos.shape[1] // natoms, 3])  # track the center of mass of each residue

    for f in range(nframes):
        for i in range(com.shape[1]):
            w = (pos[f, i * natoms:(i + 1) * natoms, :].T * mass_atoms).T  # weight each atom in the residue by its mass
            com[f, i, :] = np.sum(w, axis=0) / sum(mass_atoms)  # sum the coordinates and divide by the mass of the residue

    return com


def translate(xyz, before, after):
    """ Translate coordinates based on a reference position

    :param xyz: coordinates of set of points to be translated (n, 3)
    :param before: reference coordinate location before (3)
    :param after: reference coordinate location after (3)

    :type xyz: numpy.ndarray
    :type before: numpy.ndarray
    :type after: numpy.ndarray

    :return: translated points with respect to reference coordinate before/after locations [npts, 3]
    :rtype: numpy.ndarray
    """

    pos = np.copy(xyz)
    direction = after - before

    translation = np.array([[1, 0, 0, direction[0]], [0, 1, 0, direction[1]],
                         [0, 0, 1, direction[2]], [0, 0, 0, 1]])

    b = np.ones([1])
    for i in range(pos.shape[0]):
        coord = np.concatenate((pos[i, :], b))
        x = np.dot(translation, coord)
        pos[i, :] = x[:3]

    return pos

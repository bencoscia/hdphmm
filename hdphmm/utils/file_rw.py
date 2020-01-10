#!/usr/bin/env python

import pickle


def load_object(filename):

    with open(filename, 'rb') as f:

        return pickle.load(f)


def save_object(obj, filename):

    with open(filename, 'wb') as output:  # Overwrites any existing file.

        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def write_gro(pos, out, name='NA', box=(0., 0., 0.), ids=None, res=None, vel=None, ucell=None):
    """ write a .gro file from positions

    :param pos: xyz coordinates (natoms, 3)
    :param out: name of output .gro file
    :param name: name to give atoms being put in the .gro
    :param box: unitcell vectors. Length 9 list or length 3 list if box is cubic
    :param ids: name of each atom ordered by index (i.e. id 1 should correspond to atom 1)
    :param: res: name of residue for each atom
    :param: vel: velocity of each atom (natoms x 3 numpy array)
    :param: ucell: unit cell dimensions in mdtraj format (a 3x3 matrix)

    :type pos: np.ndarray
    :type out: str
    :type name: str
    :type box: list
    :type ids: list
    :type res: list
    :type vel: np.ndarray
    :type ucell: np.ndarray

    :return: A .gro file
    """

    if ucell is not None:
        box = [ucell[0, 0], ucell[1, 1], ucell[2, 2], ucell[0, 1], ucell[2, 0], ucell[1, 0], ucell[0, 2], ucell[1, 2],
               ucell[2, 0]]

    with open(out, 'w') as f:

        f.write('This is a .gro file\n')
        f.write('%s\n' % pos.shape[0])

        for i in range(pos.shape[0]):
            if vel is not None:
                if ids is not None:
                    f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.4f}{:8.4f}{:8.4f}\n'.format((i + 1) % 100000, '%s' % name, '%s' % name,
                                                                            (i + 1) % 100000, pos[i, 0], pos[i, 1], pos[i, 2], vel[i, 0], vel[i, 1], vel[i, 2]))
                else:
                    f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.4f}{:8.4f}{:8.4f}\n'.format((i + 1) % 100000, '%s' % res[i], '%s' % ids[i],
                                                                            (i + 1) % 100000, pos[i, 0], pos[i, 1], pos[i, 2], vel[i, 0], vel[i, 1], vel[i, 2]))

            else:
                if ids is None:
                    f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n'.format((i + 1) % 100000, '%s' % name, '%s' % name,
                                                                            (i + 1) % 100000, pos[i, 0], pos[i, 1], pos[i, 2]))
                else:
                    f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n'.format((i + 1) % 100000, '%s' % res[i], '%s' % ids[i],
                                                                            (i + 1) % 100000, pos[i, 0], pos[i, 1], pos[i, 2]))
        for i in range(len(box)):
            f.write('{:10.5f}'.format(box[i]))

        f.write('\n')
        # f.write('{:10f}{:10f}{:10f}\n'.format(0, 0, 0))
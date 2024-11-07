

# TDDMRG-CM

TDDMRG-CM is a Python program designed to make the workflow of simulating charge migration using the time-dependent density matrix renormalization group (TDDMRG) easy and straightforward. It consists of three main functionalities: ground state DMRG, application of annihilation operator, and time evolution using TDDMRG. This program is built on top of BLOCK2 (https://github.com/block-hczhai/block2-preview) and PySCF (https://github.com/pyscf/pyscf), therefore, these two programs are already be installed before using TDDMRG-CM.

To run the program, you need to prepare an input file. The input parsing environment of TDDMRG-CM has been designed so that input files are essentially a normal Python `*.py` file.  This offers a maximum flexibility for users in providing the values of input parameters to the program. Since it is an ordinary Python file, you can write the code to calculate the value of a certain input parameter right inside the input file. As an example, you want to initiate the ground state DMRG iterations from an MPS having a certain set of orbital occupancies (see `gs_occs` input definition below), and these occupancies are obtained from a separate quantum chemistry calculation. Let's say that this other calculation returns a one-particle reduced density matrix (RDM) as a matrix named `rdm.npy` in the parent folder. Then, you can give a value to the `gs_occs` input parameter in the following manner inside your input file.
```python
import numpy as np
...
rdm = np.load('../rdm.npy')
gs_occs = np.diag(rdm)
...
```
The program views the variable `rdm` as an intermediate variable, and hence will not be affected by it nor aborts even though it is not an unrecognized input variable. The downside of the above input design, however, is that any syntactical error inside the input file will not be indicated with the location where exactly it happens.

## Ground state DMRG

This task computes the ground state energy and optionally saves the final ground state MPS using DMRG algorithm. 
```python
prefix = 'H2O'
```


## Annihilation operator



## Time Evolution


## Logbook - starting a job based on a previous calculation

## Probe file


## Input parameters

##### `prefix`
abc

##### `memory`
def

##### `complex_MPS_type`








#==== General parameters ====#
<details>
  <summary><code>inp_coordinates</code></summary>
  A python multiline string that specifies the cartesian coordinates of the atoms in the molecule. The format is as follows
  
  ```
  <Atom1>  <x1>  <y1>  <z1>;
  <Atom2>  <x2>  <y2>  <z2>;
  ...
  ```
</details>

<details>
  <summary><code>inp_basis</code></summary>
  A library that specifies the atomic orbitals (AO) basis on each atom. The format follows pyscf format format for AO basis.
</details>

<details>
  <summary><code>wfn_sym</code></summary>
  The irrep of the wave function associated with the chosen value for <code>inp_symmetry</code> input. It accepts both the literal form e.g. Ag, B1, B2, A', A", as well as the integer form in PySCF notation where the trivial irrep is equal to 0. To get the complete list of the integer index equivalence of each irrep, consult the PySCF source file <code>&ltpyscf_root&gt/pyscf/symm/param.py</code>.
</details>

<details>
  <summary><code>prev_logbook</code></summary>
  The path to an existing logbook. This is used when you want to use the values of several input parameters from another simulation.
</details>

<details>
  <summary><code>nCore</code></summary>
  The number of core orbitals.
</details>

<details>
  <summary><code>nCAS</code></summary>
  The number of active orbitals.
</details>

<details>
  <summary><code>nelCAS</code></summary>
  The number of active electrons occupying the active orbitals.
</details>

<details>
  <summary><code>twos</code></summary>
  The result of 2*S where S is the total spin quantum number of the wave function.
</details>

<details>
  <summary><code>complex_MPS_type</code> (optional)</summary>
  <strong>Default</strong>: <code>'hybrid'</code>
  <br>
  The complex type of MPS in the calculation. The possible options are <code>'hybrid'</code> and <code>'full'</code>. For ground state and annihilation tasks, the choice of complex type should not matter. If they differ, then at least one of the simulations has not converged yet. For time evolution, the results will differ depending on the bond dimension. The two complex types should give identical time evolution dynamics when the bond dimension reaches convergence.
</details>

<details>
  <summary><code>dump_inputs</code> (optional)</summary>
  <strong>Default</strong>: <code>False</code>
  <br>
  If True, then the values of the input parameters will be printed to the output.
</details>

<details>
  <summary><code>memory</code> (optional)</summary>
  <strong>Default</strong>: <code>1E9</code>
  <br>
  Memory allocation in bytes for the entire run of the program.
</details>

<details>
  <summary><code>prefix</code> (optional)</summary>
  <strong>Default</strong>: The prefix of the input file if it has a <code>.py</code> extension, otherwise, the full name of the input file.
  <br>
  The prefix of files and folders created during simulation.
</details>

<details>
  <summary><code>verbose_lvl</code> (optional)</summary>
  <strong>Default</strong>: 4
  <br>
  A integer that controls the verbosity level of the output.
</details>

<details>
  <summary><code>inp_ecp</code> (optional)</summary>
  <strong>Default</strong>: <code>None</code>
  <br>
  The effective core potential (ECP) on each atom. The format follows pyscf format for ECP. If not specified, no ECP will be used.
</details>

<details>
  <summary><code>inp_symmetry</code> (optional)</summary>
  <strong>Default</strong>: <code>C1</code>
  <br>
  A string that specifies the point group symmetry of the molecule.
</details>

<details>
  <summary><code>orb_path</code> (optional)</summary>
  <strong>Default</strong>: Hartee-Fock canonical orbitals using the chosen AO basis set and geometry.
  <br>
  Specifies the site orbitals. It accepts the path to a <code>*.npy</code> file that stores a 2D array (matrix) of the AO coefficients of the orbitals, where the rows refer to the AO index and the columns refer to the orbital index. The AO used to expand the orbitals should be the same as the AO chosen for the <code>inp_basis</code> parameter. It also accepts <code>None</code>, for which case the program will treat it as if <code>orb_path</code> is not present (hence, will fall back to the default value).
</details>

<details>
  <summary><code>orb_order</code> (optional)</summary>
  <strong>Default</strong>: <code>'genetic'</code>
  <br>
  Specifies orbital ordering. The choices are as follows:
  <ol>
    <li>A string that specifies the path of a <code>*.npy</code> file containig a 1D array (vector) of integers representing the orbital index. These indices is 0-based.</li>
    <li>A list of 0-based integers. This is basically the hard-coded version of the first option above.</li>
    <li>A library of the form
      <ol type="a">
	<li><code>{'type':'linear', 'direction':(x, y, z)}</code>, or</li>
	<li><code>{'type':'circular', 'plane':&lt3x3 matrix&gt}</code></li>
      </ol>
      The a) format is for ordering based on a line in 3D space. In this ordering, the orbitals are ordered according to the projection of their dipole moments in the direction specified by the <code>'direction'</code> key. <code>x</code>, <code>y</code>, and <code>z</code> specifies the direction vector for the projection. The a) format is best used for molecules whose one of the dimenions is clearly longer than the other. The b) format is for circular ordering, best for molecules exhibiting some form of circularity in shape, e.g. aromatic molecules. The value for <code>'plane'</code> is a 3x3 numpy matrix. This matrix specifies the coordinates of three points in space with which the plane of the circular ordering is defined. The rows of this matrix correspond to the three points, while the columns correrspond to their <code>x</code>, <code>y</code>, and <code>z</code> Cartesian components.
    </li>
    <li>A string 'genetic', the genetic algorithm.</li>
    <li>A string 'fiedler', the Fiedler algorithm.</li>
  </ol>
</details>

<details>
  <summary><code>mrci</code></summary>
  <strong>Default</strong>: <code>None</code>
  <br>
  If given the format-conforming value, it prompts an MRCI calculation using MPS. The format is a library with two entries, <code>'nactive2'</code> and <code>'order'</code>. <code>'nactive2'</code> specifies the number of the excitation orbitals. <code>'nactive2':10</code> means that the last 10 orbitals of the nCAS active orbitals are considered to be the excitation orbitals. <code>'order'</code> specifies the excitation order. Currently, the available options for <code>'order'</code> is 1, 2, and 3, representing single, single-double, and single-double-triple excitations, respectively.
</details>


<!--
#==== Ground state parameters ====#
do_groundstate:
  True or False. If True, a groundstate DMRG calculation will be performed.
  D_gs:
    A list containing the schedule of the bond dimensions during DMRG
    iterations. For example, [100]*2 + [200*4] + [300], means that the first two
    iterations use a max bond dimension of 100, the next four use 200 max bond
    dimension, and beyond that it uses the max bond dimension of 300 until
    convergence or maximum iteration number is reached, whichever is earlier.
  gs_inmps_dir:
    One of the three ways to construct a guess MPS for macro iterations. If it is set to
    a valid directory path, then the guess MPS is constructed using MPS files located under
    this directory. The default of the three ways is a randomly generated MPS having the
    prescribed maximum bond dimension.
  gs_inmps_fname:
    The file name of the info file of the MPS to be used to start the ground state DMRG
    iterations. This file should be inside the folder specified through gs_inmps_dir.
    This input must be present if gs_inmps_dir is present.
  gs_noise:
    A list containing the schedule of the noise applied during ground state iterations.
    A nonzero noise can be used to prevent the MPS from getting trapped in a local
    minimum. Its format follows the same convention as D_gs.
  gs_dav_tols:
    A list containing the schedule of the tolerances to terminate the Davidson/micro
    iterations for diagonlizing the effective Hamiltonian. Typically, it starts from a
    large value such as 0.01 and decrease until e.g. 1E-7. Its format follows the same
    convention as D_gs.
  gs_steps:
    The maximum number of macro iterations in the ground state calculation. Use this
    or gs_conv_tol to determine when to terminate the macro iteration.
  gs_conv_tol:
    The energy difference tolerance when the macro iterations should stop. Use this
    or gs_steps to determine when to terminate the macro iteration.
  gs_cutoff:
    States with eigenvalue below this number will be discarded, even when
    the bond dimension is large enough to keep this state.
  gs_occs:
    One of the three ways to construct a guess MPS for macro iterations. If it is set,
    then the guess MPS is constructed in such a way that its orbital occupancies are
    equal to gs_occs. It is a vector of nCAS floating point numbers. gs_occs is
    meaningless if gs_inmps_dir is set.
  gs_bias:
    A floating point number used to shift/bias the occupancies of active orbitals
    used to construct the guess MPS for macro iterations. If gs_bias is set, the
    given initial occupancies will be modified so that high occupancies are
    reduce by an gs_bias while low occupancies are increased by gs_bias. Only
    meaningful when gs_occs is given.
  gs_outmps_dir:
    The path to the directory in which the MPS files of the final ground state
    MPS will be saved for future use.
  gs_outmps_fname:
    The file name of the info file of the final ground state MPS This input must
    be present if gs_outmps_dir is present.
  save_gs_1pdm:
    True or False. If True, the one-particle RDM of the final ground state MPS
    will be saved under gs_outmps_dir with a filename GS_1pdm.npy.
  flip_spectrum:
    True or False. If True, the macro iterations will seek the highest energy
    of the Hamiltonian. It is implemented by running the same iterations as when
    this input is False but with a -1 multiplied into the Hamiltonian.
  gs_out_cpx:
    True or False. If True, the final ground state MPS will be converted to a
    full complex MPS where the tensor elements are purely real complex numbers.
    If True and complex_MPS_type is 'full', the program will be aborted.
#==== Annihilation operation parameters ====#
do_annihilate:
  True or False. If True, the program will calculate the annihilation of an electron
  from an orbital in the given input MPS.
  ann_sp:
    True or False. The spin projection of the annihilated electron, True means alpha
    electron, otherwise, beta electron.
  ann_orb:
    Specifies which orbital from which an electron is annihilated. It accepts an
    integer ranging from 0 to nCAS-1 and a nCAS-long vector. If it is given an
    integer, the program annihilates electron from the (ann_orb+1)-th orbital of
    the site. For example, ann_orb=2 means that the an electron will be annihilated
    from the third active orbital. If ann_orb is given a vector, the program will
    annihilate an electron from the orbital represented by the linear combination
    of the site orbitals where the expansion coefficients are contained in ann_orb.
    Note that small elements of ann_orb vector can cause execution error, therefore
    user should set small elements of ann_orb vector to exactly zero before running
    the program. Usually the threshold is 1E-5, that is, in this case do
           ann_orb[np.abs(ann_orb) < 1.0E-5] = 0.0
    The final ann_orb vector must be normalized. When ann_orb is a vector, the
    irrep of orbitals with large expansion coefficients must be the same. If
    classification between large and small coefficients is not possible (e.g. due
    to low contrast of these coefficients), then set inp_symmetry to a point group
    with less symmetries. Ultimately, inp_symmetry = 'C1' should cover
    ann_orb vector of no symmetry.
  D_ann_fit:
    A list containing the schedule of the bond dimensions during the fitting
    iterations. Its format follows the same convention as D_gs.
  ann_inmps_dir:
    The path to the directory containing the MPS files of the input MPS on
    which the annihilation operator will be applied.
  ann_inmps_fname:
    The file name of the info file of the input MPS on which the annihilation
    operator will be applied. ann_inmps_fname must be located under
    ann_inmps_dir.
  ann_outmps_dir:
    The path to the directory containing the MPS files of the output MPS.
  ann_outmps_fname:
    The file name of the info file of the output MPS. ann_outmps_fname must
    be located under ann_outmps_dir.
  ann_orb_thr:
    The threshold for determining the irrep of the orbital represented by
    ann_orb in vector form. The irrep of the annihilated orbital is
    equal to the irreps of orbitals whose absolute value of coefficient
    is higher than ann_orb_thr. This implies that the irrep of these
    large-coefficient orbitals must all be the same.
  LD try:
  LD     inputs['ann_fit_margin'] = ann_fit_margin
  LD except NameError:
  LD     inputs['ann_fit_margin'] = defvals.def_ann_fit_margin
  ann_fit_noise:
    A list containing the schedule of the noise applied during fitting iterations.
    A nonzero noise can be used to prevent the MPS from getting trapped in a local
    minimum. Its format follows the same convention as D_gs.
  ann_fit_tol:
    A threshold to determine when fitting iterations should stop.
  ann_fit_steps:
    The maximum number of iteration for the fitting iterations.
  ann_fit_cutoff:
    States with eigenvalue below this number will be discarded, even when
    the bond dimension is large enough to keep this state.
  ann_fit_occs:
    If it is set, the guess MPS for fitting iterations is constructed in such a way
    that its orbital occupancies are equal to ann_fit_occs. It is a vector of nCAS
    floating point numbers.
  ann_fit_bias:
    A floating point number used to shift/bias the occupancies of active orbitals
    used to construct the guess MPS for fitting iterations. If ann_fit_bias is set,
    the given initial occupancies will be modified so that high occupancies are
    reduce by an ann_fit_bias while low occupancies are increased by ann_fit_bias.
    Only meaningful when ann_fit_occs is given.
  normalize_annout:
    True or False. If True, the output MPS after annihilation is normalized.
  save_ann_1pdm:
    True or False. If True, the one-particle RDM of the output MPS will be saved
    under ann_outmps_dir with a filename ANN_1pdm.npy.
  ann_out_singlet_embed:
    True or False. If True, the output MPS will be converted to a singlet-
    embedding representation.
  ann_out_cpx:
    True or False. If True, the final output MPS will be converted to a full
    complex MPS where the tensor elements are purely real complex numbers.
    If True and complex_MPS_type is 'full', the program will be aborted.
#==== Time evolution parameters ====#
do_timeevo:
  True or False. If True, time evolution simulation using TDDMRG will be
  performed.
  te_max_D:
    The maximum bond dimension of the time-evolving MPS in the TDDMRG
    simulation.
  tmax:
    The maximum time up to which the time evolution is run.
  dt:
    The time step for time evolution in atomic unit of time.
  tinit:
    The initial time at which the time evolution starts. It only affects
    the time points printed at which observables are calculated and printed.
    It does not affect the simulation.
  te_inmps_dir:
    The path to the directory containing the MPS files of the initial MPS
    from which the time evolution starts.
  te_inmps_fname:
    The file name of the info file of the initial MPS from which the time
    evolution starts. te_inmps_fname must be located under te_inmps_dir.
  te_inmps_cpx:
    True or False. Set it to True if the initial MPS is complex, and
    False if the initial MPS is real. When restarting a TDDMRG simulation,
    regardless of the value of complex_MPS_type, this input must be set to
    True since the last MPS from the previous TDDMRG is complex. This
    input must also be set to True if the initial MPS is not from a
    previous TDDMRG simulation but complex_MPS_type is 'full', e.g. from
    an annihilation calculation with complex_MPS_type = 'full'.
  te_inmps_multi:
    True or False. Set it to True if the initial MPS is in state-average
    format, for example, when restarting from a previous TDDMRG simulation
    where complex_MPS_type = 'hybrid'. Set it to False otherwise.
  mps_act0_dir:
    The path to the directory containing the MPS files of the MPS used as
    the state at t=0 for the computation of autocorrelation function.
  mps_act0_fname:
    The file name of the info file of the MPS used as the state at t=0
    for the computation of autocorrelation function. This file must be
    located under the mps_act0_dir directory.
  mps_act0_cpx:
    True or False. It has the same meaning as te_inmps_cpx except for
    the MPS used as the state at t=0 for the computation of autocorrelation
    function.
  mps_act0_multi:
    True or False. It has the same meaning as te_inmps_multi except for
    the MPS used as the state at t=0 for the computation of autocorrelation
    function.
  te_method:
    The time propagation method. The available options are 'rk4' and 'tdvp'.
    'rk4' is stands for the time-step targeting (TST) method, while 'tdvp'
    stands for the time-dependent variational principle method (TDVP).
  exp_tol:
  te_cutoff:
    States with eigenvalue below this number will be discarded, even when
    the bond dimension is large enough to keep this state.
  krylov_size:
    The size of Krylov subspace used to approximate the action of a matrix
    exponential on a vector in TDVP propagation. Meaningless if
    te_method = 'rk4'.
  krylov_tol:
    A threshold used to set the accuracy of the Krylov subspace method in
    approximating the action of a matrix exponential on a vector in TDVP
    propagation.
  n_sub_sweeps:
    The number of sweeps in a TST propagation used to improve the
    renormalized basis in each time step.
  n_sub_sweeps_init:
    The number of sweeps in the first time step of a TST propagation used
    to improve the renormalized basis in each time step.
  te_normalize:
    True or False. If True, the MPS will be normalized after every time
    step.
  te_sample:
    The sampling time points around which the observables will be
    calculated and printed. It accepts three formats: a numpy vector of
    monotonically increasing time points, a tuple of the form
    ('steps', n) with n an integer, and a tuple of the form ('delta', d)
    with d a float. The ('steps', n) format is used to choose sampling 
    time points using a fixed interval n. n = 1 means that the observables
    are calculated and printed exactly every time step. n = 2 means that
    the observables are calculated and printed at every second time step.
    The ('delta', d) format is used to choose sampling time points at a
    fixed time interval. d = 0.01 means that the sampling time points are
    separated by 0.01 a.u. of time.
    Note that sampling times only tell the program approximately around
    which time points should observables be calculated. The actual time
    points when the observables are printed are those determined by dt
    which are the the closest to a particular te_sample. For example, if
    the only sampling time point is 12.6 and two propagation time points
    around it is 12.0 and 13.0, then the observables will be printed at
    t = 13.0. This means that the ('steps', n) format produces sampling 
    time points that are exactly a subset of the propagation time points.
    If dt contains non-uniform time steps, however, the ('steps', n)
    format will produce sampling time points which are not uniformly
    spaced (uniform spacing might desired for Fourier transform). To
    exact subset of the propagation time points which are not uniformly
    ensure uniformly spaced sampling points that are also the spaced (as
    is usually true because the first few time steps should typically be
    really short compared to at the later times), one can do
      dt = [DT/m]*m + [DT/n]*n + [DT]
      te_sample = ('delta', p*dt[-1])
    where m, n, and p are integers, while DT is a floating point.
  te_save_mps:
    Determines how often the instantaneous MPS should be saved. The
    available options are:
      1) 'overwrite'. MPS files are saved at the sampling time points
         under the folder <prefix>.mps_t where <prefix> is the value of
         prefix input. These MPS files overwrite the MPS files saved in
         the previous sampling time point.
      2) 'sampled'. MPS files are saved at every sampling time points.
         This option can lead to a huge space taken up by the MPS files.
         This option is usually used if you want to use these
         instantaneous MPS for later analyses that are not available
         already in this program and for which the use of 1RDM alone is
         not enough. If that is not your plan, using 'overwrite\' is 
         recommended.
      3) 'no'. MPS files will not be saved.
    Regardless of the value of te_save_mps, the instantaneous MPS can be
    saved 'on-demand' by using probe files.
  te_save_1pdm:
    True or False. If True, the 1-electron RDM is saved at every
    sampling time points under the folder <prefix>.sample where <prefix>
    is the value of prefix input.
  te_save_2pdm:
  save_txt:
  save_npy:
  te_in_singlet_embed:
    A 2-entry tuple of the form (True|False, n). Specify this input
    with the first entry set to True if the initial MPS is in singlet-
    embedding format, and n (second entry) set to the actual number of
    active electrons in the system. Due to the singlet-embedded form,
    the number of electrons in the initial MPS is adjusted so that the
    total spin can be zero.
  bo_pairs:
    Lowdin bond order.
###################################################)
-->

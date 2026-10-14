/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2026 Pasteur Labs (reverse-mode AD instrumentation)
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    icoFoamADR

Description
    icoFoam with a CoDiPack reverse-mode tape over the whole time loop.

    Without -vjp this is icoFoam. With -vjp the initial velocity is registered
    as the tape input, the scalar J = sum_c w_c . U_c(T) is registered as the
    tape output, and one reverse sweep writes dJ/dU(0) per cell, which is the
    vector-Jacobian product of the final velocity field with cotangent w.

    Requires an OpenFOAM-AD build with WM_AD_MODE=ADR.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "pisoControl.H"
#include <fstream>
#include <iomanip>

#ifndef CODI_ADR
#error "icoFoamADR requires an OpenFOAM-AD build with WM_AD_MODE=ADR"
#endif

namespace
{

List<vector> readCotangent(const fileName& path, const label nCells)
{
    std::ifstream is(path);
    if (!is)
    {
        FatalErrorInFunction << "cannot open " << path << exit(FatalError);
    }

    List<vector> w(nCells, Zero);
    for (label cellI = 0; cellI < nCells; ++cellI)
    {
        double wx, wy, wz;
        if (!(is >> wx >> wy >> wz))
        {
            FatalErrorInFunction
                << "expected " << nCells << " rows in " << path
                << ", got " << cellI << exit(FatalError);
        }
        w[cellI] = vector(wx, wy, wz);
    }
    return w;
}


double peakRssMB()
{
    std::ifstream status("/proc/self/status");
    std::string key;
    double kb = 0.0;
    while (status >> key)
    {
        if (key == "VmHWM:")
        {
            status >> kb;
            break;
        }
        status.ignore(4096, '\n');
    }
    return kb/1024.0;
}

} // End anonymous namespace


int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Transient solver for incompressible laminar flow with a reverse-mode"
        " AD tape over the time loop."
    );

    argList::addBoolOption
    (
        "vjp",
        "seed the initial velocity, then write dJ/dU(0) to gradient.dat"
    );

    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    pisoControl piso(mesh);

    #include "createFields.H"
    #include "initContinuityErrs.H"

    const bool vjp = args.found("vjp");

    codi::RealReverse::Tape& tape = codi::RealReverse::getTape();

    List<vector> U0(mesh.nCells());
    forAll(U0, cellI)
    {
        U0[cellI] = U[cellI];
    }

    if (vjp)
    {
        tape.setActive();
        forAll(U0, cellI)
        {
            for (direction cmpt = 0; cmpt < 3; ++cmpt)
            {
                tape.registerInput(U0[cellI][cmpt]);
            }
        }
    }

    // Reassign through U0 so the flux carries the tape dependency, and so the
    // primal and VJP runs follow bit-identical code paths.
    forAll(U0, cellI)
    {
        U[cellI] = U0[cellI];
    }
    U.correctBoundaryConditions();
    phi = linearInterpolate(U) & mesh.Sf();

    // Usually fvc::ddtCorr is used, but we pin the coeff to 1
    // because fvc::ddtCorr applies fvcDdtPhiCoeff, whose limiter
    // makes the discrete map non-smooth along random directions.
    // Pinning the coefficient to 1 makes the map smooth
    // Caveat: forward field is 4.4% off (L2) from stock icoFoam
    // BUT the two gradients converge under grid refinement
    const dimensionedScalar rDeltaT = scalar(1)/runTime.deltaT();

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        #include "CourantNo.H"

        fvVectorMatrix UEqn
        (
            fvm::ddt(U)
          + fvm::div(phi, U)
          - fvm::laplacian(nu, U)
        );

        if (piso.momentumPredictor())
        {
            solve(UEqn == -fvc::grad(p));
        }

        while (piso.correct())
        {
            volScalarField rAU(1.0/UEqn.A());
            volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
            surfaceScalarField phiHbyA
            (
                "phiHbyA",
                fvc::flux(HbyA)
              + fvc::interpolate(rAU)*rDeltaT
               *(phi.oldTime() - fvc::dotInterpolate(mesh.Sf(), U.oldTime()))
            );

            adjustPhi(phiHbyA, U, p);

            constrainPressure(p, U, phiHbyA, rAU);

            while (piso.correctNonOrthogonal())
            {
                fvScalarMatrix pEqn
                (
                    fvm::laplacian(rAU, p) == fvc::div(phiHbyA)
                );

                pEqn.setReference(pRefCell, pRefValue);

                pEqn.solve(p.select(piso.finalInnerIter()));

                if (piso.finalNonOrthogonalIter())
                {
                    phi = phiHbyA - pEqn.flux();
                }
            }

            #include "continuityErrs.H"

            U = HbyA - rAU*fvc::grad(p);
            U.correctBoundaryConditions();
        }

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    if (vjp)
    {
        const List<vector> w =
            readCotangent(runTime.path()/"cotangent.dat", mesh.nCells());

        scalar J = 0.0;
        forAll(U, cellI)
        {
            for (direction cmpt = 0; cmpt < 3; ++cmpt)
            {
                J += w[cellI][cmpt]*U[cellI][cmpt];
            }
        }
        reduce(J, sumOp<scalar>());

        tape.registerOutput(J);
        tape.setPassive();
        J.setGradient(1.0);
        tape.evaluate();

        std::ofstream os((runTime.path()/"gradient.dat").c_str());
        os << std::setprecision(17);
        forAll(U0, cellI)
        {
            for (direction cmpt = 0; cmpt < 3; ++cmpt)
            {
                os << U0[cellI][cmpt].getGradient() << "\n";
            }
        }
    }

    Info<< "peakRssMB " << peakRssMB() << endl;
    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //

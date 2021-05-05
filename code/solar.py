# This file implements the classes for solar systems and space missions

import numpy as np
import matplotlib.pyplot as plt
from rksolvers import rk4, planet_derivs, mission_derivs


class Planet:

    """
    """

    def __init__(self, r, v, mass, name):
        self.name = name
        self.mass = mass
        self.r = np.copy(r)
        self.v = np.copy(v)
        self.r0 = np.copy(r)
        self.v0 = np.copy(v)
        self.xplot = np.array([])
        self.yplot = np.array([])

    def update(self, r, v):
        self.r = np.copy(r)
        self.v = np.copy(v)
        self.xplot = np.append(self.xplot,r[0])
        self.yplot = np.append(self.yplot,r[1])

    def random_step(self, tau, period, GM=4*np.pi**2):
        time = 0.0
        steps = int(np.random.rand() * period / tau)
        for _ in range(steps):
            state = np.array([self.mass,self.r[0],self.r[1],self.v[0],self.v[1]])
            state = rk4(state,time,tau,planet_derivs, GM=GM)
            self.r = np.array([state[1],state[2]])
            self.v = np.array([state[3],state[4]])
            time += tau
        self.r0 = np.copy(self.r)
        self.v0 = np.copy(self.v)

    def reset(self):
        self.r = np.copy(self.r0)
        self.v = np.copy(self.v0)
        self.xplot = np.array([])
        self.yplot = np.array([])


class Rocket:

    """
    """

    def __init__(self, r, v):
        self.name = "Rocket"
        self.mass = 3.68e-28 # mass of Voyager 1 in AMU
        self.r = np.copy(r)
        self.v = np.copy(v)
        self.r0 = np.copy(r)
        self.v0 = np.copy(v)
        self.xplot = np.array(r[0])
        self.yplot = np.array(r[1])
        self.total_dv = 0.0
        self.dvplot = np.array([0.0])
        self.tdvplot = np.array([0.0])

    def update(self, r, v, action):
        self.r = np.copy(r)
        self.v = np.copy(v) + action
        self.xplot = np.append(self.xplot,r[0])
        self.yplot = np.append(self.yplot,r[1])
        deltaV = np.linalg.norm(action)
        self.total_dv += deltaV
        self.dvplot = np.append(self.dvplot,deltaV)
        self.tdvplot = np.append(self.tdvplot,self.total_dv)

    def plot(self, title='', file=''):
        plt.figure(1); plt.clf()
        plt.plot(self.dvplot,'.',label='delta')
        plt.plot(self.tdvplot,'-',label='total')
        plt.xlabel('Step Number')
        plt.ylabel('dV (AU/yr)')
        plt.legend()
        plt.title(title)
        plt.axis('tight')
        plt.axis('equal')
        plt.grid(True)
        if file: plt.savefig(file)
        else: plt.show()

    def reset(self):
        self.r = np.copy(self.r0)
        self.v = np.copy(self.v0)
        self.xplot = np.array(self.r[0])
        self.yplot = np.array(self.r[1])
        self.total_dv = 0.0
        self.dvplot = np.array([0.0])
        self.tdvplot = np.array([0.0])


class SolarSystem:

    def __init__(self, tau, planets, GM=4*np.pi**2):
        self.planets = planets
        self.time = 0
        self.tau = tau
        self.GM = GM

    def step(self):
        state = []
        for p in self.planets: state.append([p.mass, p.r[0], p.r[1], p.v[0], p.v[1]])
        new = rk4(np.array(state), self.time, self.tau, mission_derivs, GM=self.GM)
        for i,p in enumerate(self.planets):
            p.update(np.array([new[i][1], new[i][2]]), np.array([new[i][3], new[i][4]]))

    def plot(self, title='', file=''):
        # Graph the trajectories of the planets
        plt.figure(0); plt.clf()
        plt.plot(0,0,'yo',label='Sun')
        for planet in self.planets:
            plt.plot(planet.xplot,planet.yplot,'-',label=planet.name)
        plt.xlabel('x (AU)')
        plt.ylabel('y (AU)')
        plt.legend()
        plt.title(title)
        plt.axis('tight')
        plt.axis('equal')
        plt.grid(True)
        if file: plt.savefig(file)
        else: plt.show()


class SimpleHighThrustMission:

    """
    """

    def __init__(self, tau, source, target, rocket, GM=4*np.pi**2):
        self.tau = tau
        self.time = 0.0
        self.step_count = 0
        self.rocket = rocket
        self.source = source
        self.target = target
        self.target_tolerance = 0.02
        self.min_dist = np.linalg.norm(rocket.r - target.r)
        self.GM = GM

    def step(self, action):
        # self.rocket.boost(action) # implement instaneous change in velocity
        # runs a 4th order runga-kutta to move the planets and the rocket
        new = rk4(self.state(), self.time, self.tau, mission_derivs, GM=self.GM)
        # updates the position and velocity of rocket (cost updates too), target, source, and planets
        self.rocket.update(np.array([new[0][1], new[0][2]]), np.array([new[0][3], new[0][4]]), action)
        self.source.update(np.array([new[1][1], new[1][2]]), np.array([new[1][3], new[1][4]]))
        self.target.update(np.array([new[2][1], new[2][2]]), np.array([new[2][3], new[2][4]]))
        # checks if the rockets has gotten new min dist to target and update if so
        if np.linalg.norm(self.rocket.r - self.target.r) < self.min_dist:
            self.min_dist = np.linalg.norm(self.rocket.r - self.target.r)
        self.time += self.tau; self.step_count += 1  # # update the step and time

    def state(self):
        # create a state vector for the rocket and planets
        # rocket has 0 index, source has 1, target has 2
        state = [[self.rocket.mass, self.rocket.r[0], self.rocket.r[1], self.rocket.v[0], self.rocket.v[1]]]
        state.append([self.source.mass, self.source.r[0], self.source.r[1], self.source.v[0], self.source.v[1]])
        state.append([self.target.mass, self.target.r[0], self.target.r[1], self.target.v[0], self.target.v[1]])
        return np.array(state)

    def observation(self):
        return [item for sublist in self.state() for item in sublist]

    def done(self):
        if self.min_dist <= self.target_tolerance: return True
        return False

    def plot_mission(self, title='', file=''):
        # Create graph of the trajectories of the planets
        # If a file is given, then the graph is stored at that location
        # Otherwise, the graph is shown locally
        plt.figure(0); plt.clf()
        plt.plot(0,0,'yo',label='Sun')
        plt.plot(self.source.xplot,self.source.yplot,'r-',label='Source')
        plt.plot(self.source.r[0],self.source.r[1],'r.')
        plt.plot(self.target.xplot,self.target.yplot,'g-',label='Target')
        plt.plot(self.target.r[0],self.target.r[1],'g.')
        plt.plot(self.rocket.xplot, self.rocket.yplot,'b-',label='Rocket')
        plt.plot(self.rocket.r[0],self.rocket.r[1],'b.')
        plt.xlabel('x (AU)')
        plt.ylabel('y (AU)')
        plt.legend()
        plt.title(title)
        plt.axis('tight')
        plt.axis('equal')
        plt.grid(True)
        if file: plt.savefig(file)
        else: plt.show()

    def plot_rocket(self, title='', file=''):
        self.rocket.plot(title=title, file=file)

    def plot_reward(self, title='', file=""):
        pass

    def plot_all(self):
        pass

    def reward(self):
        return -self.min_dist-self.rocket.total_dv

    def reset(self):
        self.time = 0.0
        self.step_count = 0
        self.rocket.reset()
        self.source.reset()
        self.target.reset()
        self.min_dist = np.linalg.norm(self.rocket.r - self.target.r)


class ComplexHighThrustMission(SimpleHighThrustMission):

    """
    """

    def __init__(self, tau, source, target, rocket, planets, GM=4*np.pi**2):
        super().__init__(tau, source, target, rocket, GM=GM)
        self.planets = planets

    def step(self, action):
        """
        """
        # runs a 4th order runga-kutta to move the planets and the rocket
        new = rk4(self.state(), self.time, self.tau, mission_derivs, GM=self.GM)
        # updates the position and velocity of rocket (cost updates too), target, source, and planets
        self.rocket.update(np.array([new[0][1], new[0][2]]), np.array([new[0][3], new[0][4]]), action)
        self.source.update(np.array([new[1][1], new[1][2]]), np.array([new[1][3], new[1][4]]))
        self.target.update(np.array([new[2][1], new[2][2]]), np.array([new[2][3], new[2][4]]))
        for i,p in enumerate(self.planets):
            p.update(np.array([new[i+3][1], new[i+3][2]]), np.array([new[i+3][3], new[i+3][4]]))
        # checks if the rockets has gotten new min dist to target and update if so
        if np.linalg.norm(self.rocket.r - self.target.r) < self.min_dist:
            self.min_dist = np.linalg.norm(self.rocket.r - self.target.r)
        self.time += self.tau; self.step_count += 1  # # update the step and time

    def state(self):
        # creates a state vector for the rocket and planets
        # rocket always holds 0 index, source fills 1, target fills 2, and other planets fill rest
        state = [[self.rocket.mass, self.rocket.r[0], self.rocket.r[1], self.rocket.v[0], self.rocket.v[1]]]
        state.append([self.source.mass, self.source.r[0], self.source.r[1], self.source.v[0], self.source.v[1]])
        state.append([self.target.mass, self.target.r[0], self.target.r[1], self.target.v[0], self.target.v[1]])
        for p in self.planets: state.append([p.mass, p.r[0], p.r[1], p.v[0], p.v[1]])
        return np.array(state)

    def plot_mission(self, title='', file=''):
        # Create graph of the trajectories of the planets
        # If a file is given, then the graph is stored at that location
        # Otherwise, the graph is shown locally
        plt.figure(0); plt.clf()
        plt.plot(0,0,'yo',label='Sun')
        plt.plot(self.source.xplot,self.source.yplot,'r-',label='Source')
        plt.plot(self.source.r[0],self.source.r[1],'r.')
        plt.plot(self.target.xplot,self.target.yplot,'g-',label='Target')
        plt.plot(self.target.r[0],self.target.r[1],'g.')
        plt.plot(self.rocket.xplot, self.rocket.yplot,'b-',label='Rocket')
        plt.plot(self.rocket.r[0],self.rocket.r[1],'b.')
        for p in self.planets:
            plt.plot(p.xplot, p.yplot,'c-',)
            plt.plot(p.r[0],p.r[1],'c.')
        plt.xlabel('x (AU)')
        plt.ylabel('y (AU)')
        plt.legend()
        plt.title(title)
        plt.axis('tight')
        plt.axis('equal')
        plt.grid(True)
        if file: plt.savefig(file)
        else: plt.show()

    def reset(self):
        self.rocket.reset()
        self.source.reset()
        self.target.reset()
        for p in self.planets:
            p.reset()


def rotate_vector(r,theta):
    newx = r[0]*np.cos(theta) - r[1]*np.sin(theta)
    newy = r[0]*np.sin(theta) + r[1]*np.cos(theta)
    return np.array([newx,newy])


def CreateRandomPlanet(tau, minr, name=''):
    r0 = (np.random.rand() * (5 - minr)) + minr
    # mass must fall btwn approx those of jupyter and pluto in solar mass units
    min_mass = 6.75e-9; max_mass = 0.001
    mass = (np.random.rand() * (max_mass-min_mass)) + min_mass
    # velocity at the aphelion is given by: v = sqrt(GM/a*(1-e)/(1+e))
    eccentricity = np.random.rand() * 0.85
    v0 = np.sqrt(4*(np.pi**2)*(1-eccentricity) / r0)
    period = (r0 / (1+eccentricity))**3
    # rotate the orbit a random angle
    angle = 2*np.pi*np.random.rand()
    r = rotate_vector(np.array([r0, 0.]), angle)
    v = rotate_vector(np.array([0., v0]), angle)
    planet = Planet(r, v, mass, name)
    # move planet random number of steps so don't always start at aphelion
    planet.random_step(tau,period)
    return planet


def CreateRandomSolarSystem(tau):
    source = CreateRandomPlanet(tau, 0.5, name='Source')
    target = CreateRandomPlanet(tau, np.linalg.norm(source.r), name='Target')
    return SolarSystem(tau, [source, target])


def CreateRandomSimpleHighThrustMission(tau):
    source = CreateRandomPlanet(tau, 0.5, name='Source')
    target = CreateRandomPlanet(tau, np.linalg.norm(source.r), name='Target')
    rocket = Rocket(source.r, source.v)
    return SimpleHighThrustMission(tau, source, target, rocket)


def CreateRandomComplexHighThrustMission(tau, num_planets):
    source = CreateRandomPlanet(tau, 0.5, name='Source')
    target = CreateRandomPlanet(tau, np.linalg.norm(source.r), name='Target')
    rocket = Rocket(source.r, source.v)
    planets = [CreateRandomPlanet(tau, 0.5) for _ in range(num_planets)]
    return ComplexHighThrustMission(tau, source, target, rocket, planets)

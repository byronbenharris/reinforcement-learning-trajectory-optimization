# This file implements the classes for solar systems and space missions

import numpy as np
import matplotlib.pyplot as plt
from rksolvers import rk4, planet_derivs, mission_derivs

# Random num generator for all solar classes
RNG = np.random.default_rng(42)

class Planet:

    """
    The planet class contains all the core information and frequently used
    methods. This is the base object that composes the SolarSystem and Missions.
    """

    def __init__(self, r, v, mass, name, eccentricity, aphelion, period):

        self.name = name
        self.mass = mass
        self.r = np.copy(r)
        self.v = np.copy(v)
        self.r0 = np.copy(r)
        self.v0 = np.copy(v)
        self.xplot = np.array([])
        self.yplot = np.array([])
        self.eccentricity = eccentricity
        self.aphelion = aphelion
        self.period = period

    def update(self, r, v):
        self.r = np.copy(r)
        self.v = np.copy(v)
        self.xplot = np.append(self.xplot,r[0])
        self.yplot = np.append(self.yplot,r[1])

    def random_step(self, tau, sun_mass=1):
        # this moves the planet a random number of steps less than the period
        # so that the planet will start at a random place in the orbit
        time = 0.0
        steps = int(RNG.random() * self.period / tau)
        for _ in range(steps):
            state = np.array([self.mass,self.r[0],self.r[1],self.v[0],self.v[1]])
            state = rk4(state,time,tau,planet_derivs, sun_mass=sun_mass)
            self.r = np.array([state[1],state[2]])
            self.v = np.array([state[3],state[4]])
            time += tau
        self.r0 = np.copy(self.r)
        self.v0 = np.copy(self.v)

    def reset(self):
        # planet to loc and vel
        self.r = np.copy(self.r0)
        self.v = np.copy(self.v0)
        self.xplot = np.array([])
        self.yplot = np.array([])


class Rocket:

    """
    The rocket class implements a High-Thrust Rocket. This means that the rocket
    has no limitation of acceleration (ie. velocity is instantly implemented --
    see `update`).

    This change is critical to keep this problem from getting really really
    complex really fast.
    """

    def __init__(self, r, v):

        self.name = "Rocket"
        # mass of Voyager 1 in AMU
        self.mass = 3.68e-28
        self.r = np.copy(r)
        self.v = np.copy(v)
        self.r0 = np.copy(r)
        self.v0 = np.copy(v)
        self.dv = 1.0
        self.dv_sum = 0.0

        self.xplot = np.array(r[0])
        self.yplot = np.array(r[1])

    def update(self, r, v, action):
        self.r = np.copy(r)
        self.v = np.copy(v) + self.boost(action)
        self.xplot = np.append(self.xplot,r[0])
        self.yplot = np.append(self.yplot,r[1])

    def boost(self, action):
        if action == 0: return np.array([0, 0])
        self.dv_sum += self.dv
        theta = 2 * np.pi * (action-1) / 16
        return rotate_vector(np.array([self.dv,0.0]), theta)

    def reset(self):
        # resets to rocket to the original state
        self.r = np.copy(self.r0)
        self.v = np.copy(self.v0)
        self.xplot = np.array(self.r[0])
        self.yplot = np.array(self.r[1])
        self.dv_sum = 0.0
        self.dvplot = np.array([0.0])
        self.tdvplot = np.array([0.0])


class SolarSystem:

    """
    The solar system is composed of a list of planets with a sun (1 AMU)
    stationary at the origin. This class is primarily used for testing step
    methods (ie rka, rk4).
    """

    def __init__(self, tau, planets, sun_mass=1):
        self.planets = planets
        self.time = 0
        self.tau = tau
        self.sun_mass = 1

    def step(self):
        state = []
        for p in self.planets: state.append([p.mass, p.r[0], p.r[1], p.v[0], p.v[1]])
        new = rk4(np.array(state), self.time, self.tau, mission_derivs, sun_mass=self.sun_mass)
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
    The SimpleHighThrustMission is a class for managing tranjectory analysis for
    a high thrust rocket. The rocket starts on planet `source` and aims to travel
    to planet `target`.
    """

    def __init__(self, tau, source, target, rocket):
        self.tau = tau
        self.time = 0.0
        self.step_count = 0
        self.rocket = rocket
        self.source = source
        self.target = target
        self.target_tolerance = 0.01
        self.dist = np.linalg.norm(rocket.r - target.r)
        self.dist0 = np.linalg.norm(rocket.r - target.r)
        self.min_dist = self.dist
        self.max_planet_dist = max(source.aphelion, target.aphelion)
        self.sun_mass = 1

    def step(self, action):
        # self.rocket.boost(action) # implement instaneous change in velocity
        # runs a 4th order runga-kutta to move the planets and the rocket
        new = rk4(self.state(), self.time, self.tau, mission_derivs, sun_mass=self.sun_mass)
        # updates the position and velocity of rocket (cost updates too), target, source, and planets
        self.rocket.update(np.array([new[0][1], new[0][2]]), np.array([new[0][3], new[0][4]]), action)
        self.source.update(np.array([new[1][1], new[1][2]]), np.array([new[1][3], new[1][4]]))
        self.target.update(np.array([new[2][1], new[2][2]]), np.array([new[2][3], new[2][4]]))
        self.dist = np.linalg.norm(self.rocket.r - self.target.r)
        # checks if the rockets has gotten new min dist to target and update if so
        if self.dist < self.min_dist: self.min_dist = self.dist
        self.time += self.tau; self.step_count += 1  # update the step and time

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
        # if the rocket has hit the target, done :)
        if self.dist <= self.target_tolerance: self.dist = 0; return True
        # if the rocket has gone at least two times farthest planet, done :(
        if np.linalg.norm(self.rocket.r) > 3*self.max_planet_dist: return True
        return False # otherwise, keep going

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

    def reward(self):
        return 1000*(1-(self.dist/self.dist0)) - self.rocket.dv_sum

    def reset(self):
        # resets mission to original state
        self.time = 0.0
        self.step_count = 0
        self.rocket.reset()
        self.source.reset()
        self.target.reset()
        self.dist = np.linalg.norm(self.rocket.r - self.target.r)
        self.min_dist = self.dist


class ComplexHighThrustMission(SimpleHighThrustMission):

    """
    Basically the same as a SimpleHighThrustMission except that in addition to the
    source and target their are an additional n planets in the system. Otherwise
    works the same.
    """

    def __init__(self, tau, source, target, rocket, planets, sun_mass=1):
        super().__init__(tau, source, target, rocket, sun_mass=sun_mass)
        self.planets = planets
        self.max_planet_dist = max(max(source.aphelion, target.aphelion),
            max([p.aphelion for p in planets]))

    def step(self, action):
        # runs a 4th order runga-kutta to move the planets and the rocket
        new = rk4(self.state(), self.time, self.tau, mission_derivs, sun_mass=self.sun_mass)
        # updates the position and velocity of rocket (cost updates too), target, source, and planets
        self.rocket.update(np.array([new[0][1], new[0][2]]), np.array([new[0][3], new[0][4]]), action)
        self.source.update(np.array([new[1][1], new[1][2]]), np.array([new[1][3], new[1][4]]))
        self.target.update(np.array([new[2][1], new[2][2]]), np.array([new[2][3], new[2][4]]))
        for i,p in enumerate(self.planets):
            p.update(np.array([new[i+3][1], new[i+3][2]]), np.array([new[i+3][3], new[i+3][4]]))
        self.dist = np.linalg.norm(self.rocket.r - self.target.r)
        # checks if the rockets has gotten new min dist to target and update if so
        if self.dist < self.min_dist: self.min_dist = self.dist
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
        # resets mission to original state
        # super().reset()
        self.time = 0.0
        self.step_count = 0
        self.rocket.reset()
        self.source.reset()
        self.target.reset()
        self.dist = np.linalg.norm(self.rocket.r - self.target.r)
        self.min_dist = self.dist
        for p in self.planets:
            p.reset()


def rotate_vector(r,theta):
    # rotates vector `r` by `theta` radians
    newx = r[0]*np.cos(theta) - r[1]*np.sin(theta)
    newy = r[0]*np.sin(theta) + r[1]*np.cos(theta)
    return np.array([newx,newy])


def CreateRandomPlanet(tau, minr, name=''):
    # creates a new and completely random planet
    r0 = (RNG.random() * (5 - minr)) + minr
    # mass must fall btwn approx those of jupyter and pluto in solar mass units
    # min_mass = 6.75e-9; max_mass = 0.001
    mass = 1.0e-5 # (RNG.random() * (max_mass-min_mass)) + min_mass
    # velocity at the aphelion is given by: v = sqrt(GM/a*(1-e)/(1+e))
    eccentricity = RNG.random() * 0.85
    v0 = np.sqrt(4*(np.pi**2)*(1-eccentricity) / r0)
    period = (r0 / (1+eccentricity))**3
    aphelion = r0*(1+eccentricity)
    # rotate the orbit a random angle
    angle = 2*np.pi*RNG.random()
    r = rotate_vector(np.array([aphelion, 0.]), angle)
    v = rotate_vector(np.array([0., v0]), angle)
    planet = Planet(r, v, mass, name, eccentricity, aphelion, period)
    # move planet random number of steps so don't always start at aphelion
    planet.random_step(tau)
    return planet


def CreateRandomSolarSystem(tau):
    # creates a new and completely random solar system
    source = CreateRandomPlanet(tau, 0.5, name='Source')
    target = CreateRandomPlanet(tau, np.linalg.norm(source.r), name='Target')
    return SolarSystem(tau, [source, target])


def CreateRandomSimpleHighThrustMission(tau):
    # creates a new and completely random simple high thrust mission
    source = CreateRandomPlanet(tau, 0.5, name='Source')
    target = CreateRandomPlanet(tau, np.linalg.norm(source.r), name='Target')
    rocket = Rocket(source.r, source.v)
    return SimpleHighThrustMission(tau, source, target, rocket)


def CreateRandomComplexHighThrustMission(tau, num_planets):
    # creates a new and completely random complex high thrust mission
    source = CreateRandomPlanet(tau, 0.5, name='Source')
    target = CreateRandomPlanet(tau, np.linalg.norm(source.r), name='Target')
    rocket = Rocket(source.r, source.v)
    planets = [CreateRandomPlanet(tau, 0.5) for _ in range(num_planets)]
    return ComplexHighThrustMission(tau, source, target, rocket, planets)

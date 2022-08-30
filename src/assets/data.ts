interface Project {
  name: string;
  technologies: string[];
  description: string;
  website?: string;
}

interface Experience {
  as: string;
  company: string;
  website?: string;
  year: string[];
}

export const dataExperiences: Experience[] = [
  {
    company: 'Skourse',
    as: 'Instructor',
    year: ['2022'],
    website: 'https://www.skourse.com/',
  },
  {
    company: 'Symph',
    as: 'Full Stack Developer',
    year: ['2020', '2022'],
    website: 'https://www.symph.co/',
  },
  {
    company: 'Redtomato Design Studio',
    as: 'Software Engineer',
    year: ['2018', '2020'],
  },
  {
    company: 'UnionBank of the Philippines',
    as: 'Software Engineer Intern',
    year: ['2016'],
    website: 'https://unionbankph.com/',
  },
];

export const dataProjects: Project[] = [
  {
    name: 'EFOI',
    technologies: ['Python2', 'GCP', 'AppEngine', 'Datastore', 'Javascript'],
    description:
      'Electronic Freedom of Information Website and Platform - Implemented features’ improvements and adjustments both backend and frontend of the application',
    website: 'https://www.foi.gov.ph/',
  },
  {
    name: 'GivingSide',
    technologies: ['Postgres', 'Ruby on Rails', 'Heroku'],
    description:
      'Built a pipeline for automating parsing texts with regular expressions',
    website: 'https://www.givingside.com/',
  },
  {
    name: 'Sentimental Analysis',
    technologies: ['AWS Sagemaker', 'Python', 'Pytorch'],
    description:
      'Built a straightforward recurrent neural network, to determine the sentiment of a movie review using the IMDB data set',
  },
  {
    name: 'Udagram',
    technologies: ['Typescript', 'AWS', 'Node JS', 'React JS', 'DynamoDBS'],
    description:
      'Refactored Udagram, a cloud application developed alongside the Udacity Cloud Engineering Nanodegree. It allows users to register and log into a web client, post photos to the feed, and process photos, to different types of architectures - from monolith to microservices, to serverless',
  },
];

export const resumeLink =
  'https://docs.google.com/document/d/1-BoDh5MLDDqqb_qIKv3gd7IMSogKHstWb66jpdNHRgk/edit?usp=sharing';

export const linkedinLink = 'https://www.linkedin.com/in/rachemoniq/';

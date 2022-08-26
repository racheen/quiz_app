import Card from '../../components/Card';
import {
  H1,
  H2,
  H4,
  Intro1,
  Intro2,
  Section,
  SectionContent,
  SectionText,
  SectionTitle,
} from './styles';
import { dataExperiences, dataProjects } from '../../assets/data';

export default function HomePage() {
  const projects = dataProjects;
  const experiences = dataExperiences;
  const MAX_DISPLAY = 3;

  return (
    <>
      <Section full={true} rightOnly={true}>
        <Intro1>
          <H4>Hi, I'm</H4>
          <H1>&nbsp;Rachem</H1>
        </Intro1>
        <Intro2>
          <H2>I code and illustrate with my cats</H2>
        </Intro2>
      </Section>
      <Section both={true}>
        <SectionTitle>to get started...</SectionTitle>
        <SectionText>
          I am an aspiring machine learning engineer, co-founder of a startup
          company - Redtomato Design Studio.
        </SectionText>
        <SectionText>
          My interest in programming started back in 2012 when I was doing
          animation using Adobe Flash. After that year, I got exposed to basic
          programming with Visual Basic that I decided to take a computer
          science bachelor’s degree.
        </SectionText>
        <SectionText>
          I also take a break from developing and building with codes - I do
          water painting and digital drawing~
        </SectionText>
      </Section>
      <Section>
        <SectionTitle>where I've worked...</SectionTitle>
        <SectionContent>
          {experiences.slice(0, MAX_DISPLAY).map((experience) => (
            <Card
              company={experience.company}
              subcontent={experience.year}
              website={experience.website}
              workTitle={experience.as}
            />
          ))}
        </SectionContent>
      </Section>
      <Section>
        <SectionTitle>some notable projects...</SectionTitle>
        <SectionContent>
          {projects.slice(0, MAX_DISPLAY).map((project) => (
            <Card
              content={project.description}
              isProject={true}
              subcontent={project.technologies}
              title={project.name}
              website={project.website}
            />
          ))}
        </SectionContent>
      </Section>
    </>
  );
}

import {
  Container,
  Content,
  LightGreenText,
  MediumGrayText,
  Subcontent,
  Title,
} from './styles';

type Props = {
  company?: string;
  content?: string;
  isProject?: boolean;
  subcontent?: string[];
  title?: string;
  workTitle?: string;
  website?: string;
};

export default function Card(props: Props) {
  const { content, company, isProject, subcontent, title, website, workTitle } =
    props;
  return (
    <Container isProject={isProject}>
      <a href={website} target='_blank' rel='noreferrer'>
        {isProject && (
          <>
            <Title>{title}</Title>
            <Content>{content}</Content>
            <Subcontent>{subcontent?.join(' | ')}</Subcontent>
          </>
        )}
        {!isProject && (
          <>
            <Title>
              {company} <LightGreenText> as </LightGreenText>{' '}
              <MediumGrayText>{workTitle}</MediumGrayText>
            </Title>
            <Subcontent>{subcontent?.join('-')}</Subcontent>
          </>
        )}
      </a>
    </Container>
  );
}

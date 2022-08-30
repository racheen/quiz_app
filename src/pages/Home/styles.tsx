import styled, { css } from 'styled-components';

type SectionProps = {
  full?: boolean;
  rightOnly?: boolean;
  both?: boolean;
};

export const Section = styled.div<SectionProps>`
  color: ${(props) => props.theme.colors.darkGray};
  padding-top: 50px;
  padding-bottom: 50px;
  min-height: 300px;

  ${(props) =>
    props.full &&
    css`
      padding-top: 0;
      min-height: 100vh;
      justify-content: center;
      align-items: center;
    `}
  ${(props) =>
    props.both &&
    css`
      background: linear-gradient(${props.theme.colors.lightGreen} 0 0)
          calc(0.25 * 100% / 8),
        linear-gradient(${props.theme.colors.lightGreen} 0 0)
          calc(7.5 * 100% / 8);
      background-size: 10px 100%;
      background-repeat: no-repeat;
    `}
    ${(props) =>
    props.rightOnly &&
    css`
      background: linear-gradient(${props.theme.colors.lightGreen} 0 0)
        calc(7.5 * 100% / 8);
      background-size: 10px 90%;
      background-repeat: no-repeat;
    `};
`;

export const SectionContentLayout = styled.div`
  display: flex;
  padding: 50px 100px 0 100px;
`;

export const SectionContent = styled(SectionContentLayout)`
  flex-wrap: wrap;
  justify-content: space-evenly;
`;

export const TextDefault = styled(SectionContentLayout)`
  align-items: baseline;
  padding-top: 20px;
`;

export const Intro1 = styled(TextDefault)`
  padding: 250px 100px 0 100px;
  width: 586px;
  color: ${(props) => props.theme.colors.mediumGray};
  font-weight: ${(props) => props.theme.fontWeights.bold};
`;

export const Intro2 = styled(Intro1)`
  display: block;
  padding: 0 0 0 100px;
`;

export const H1 = styled.div`
  font-size: ${(props) => props.theme.fontSizes.h1};
  color: ${(props) => props.theme.colors.lightGreen};
`;

export const H4 = styled.div`
  font-size: ${(props) => props.theme.fontSizes.h4};
`;

export const H2 = styled.div`
  font-size: ${(props) => props.theme.fontSizes.h2};
  color: ${(props) => props.theme.colors.mediumGray};
`;

export const SectionTitle = styled(TextDefault)`
  font-size: ${(props) => props.theme.fontSizes.h4};
  color: ${(props) => props.theme.colors.lightGreen};
  font-weight: ${(props) => props.theme.fontWeights.bold};
`;

export const SectionText = styled(TextDefault)`
  font-weight: ${(props) => props.theme.fontWeights.mediumLight};
  font-size: ${(props) => props.theme.fontSizes.medium};
  color: ${(props) => props.theme.colors.darkGray};
  line-height: ${(props) => props.theme.fontSizes.h3};
  width: 700px;
`;

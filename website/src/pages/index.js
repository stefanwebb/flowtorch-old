import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

// Prism (Rust)
import Prism from "prism-react-renderer/prism";
(typeof global !== "undefined" ? global : window).Prism = Prism;
require("prismjs/components/prism-rust");

// Our theme
import Examples from "@theme/Examples";
import Features from "@theme/Features";

function Home() {
  const context = useDocusaurusContext();
  const {siteConfig = {}} = context;
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <div className="row">
            <div className="col col--5 col--offset-1">
          <h1 className="hero__title">{siteConfig.title}</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className={clsx(
                'button button--outline button--secondary button--lg',
                styles.getStarted,
              )}
              to={useBaseUrl('users')}>
              Get Started
            </Link>
          </div>
          </div>
          <div className="col col--5">
          <img className={styles.heroImg} src="img/logo.svg" />
          </div>
          </div>
        </div>
      </header>
      <main>
          <Features />
          <Examples />
      </main>
    </Layout>
  );
}

export default Home;

import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Search from "./pages/Search";
import Ingest from "./pages/Ingest";
import Manage from "./pages/Manage";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/search" replace />} />
        <Route path="/search" element={<Search />} />
        <Route path="/ingest" element={<Ingest />} />
        <Route path="/manage" element={<Manage />} />
      </Routes>
    </Layout>
  );
}

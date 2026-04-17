import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Search from "./pages/Search";
import Ingest from "./pages/Ingest";
import Manage from "./pages/Manage";
import Viewer from "./pages/Viewer";

export default function App() {
  return (
    <Routes>
      {/* Page viewer — full width, no sidebar */}
      <Route path="/view/:hash/:page" element={<Viewer />} />

      {/* Main app pages — sidebar layout */}
      <Route path="/*" element={
        <Layout>
          <Routes>
            <Route path="/" element={<Navigate to="/search" replace />} />
            <Route path="/search" element={<Search />} />
            <Route path="/ingest" element={<Ingest />} />
            <Route path="/manage" element={<Manage />} />
          </Routes>
        </Layout>
      } />
    </Routes>
  );
}

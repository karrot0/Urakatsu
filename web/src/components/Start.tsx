import { Boxes, Link2, Plus, Search, Settings, Trash2, X } from "lucide-react";
import { useMemo, useState } from "react";

import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { sources } from "../api/sources";
import { formatDay, formatTimestamp, plural } from "../format";
import { useWorkspace } from "../state/workspace";
import { ThemeSwitch } from "./CommandBar";
import { TitleBarInset } from "./WindowControls";
import { Select } from "./primitives";
import { Badge } from "./ui";
import { cn } from "cn";

type ProjectSort = "updated" | "name";
const SORTS: { value: ProjectSort; label: string }[] = [
  { value: "updated", label: "Recently updated" },
  { value: "name", label: "Name" },
];

const EMPTY_BLOCK = "flex flex-col items-center gap-0.5 rounded-card border border-dashed border-line p-8 text-center";
const COVER = "aspect-[5/7] mb-1 w-full rounded-control bg-surface-fill object-cover shadow-raised";

export function Start() {
  const ws = useWorkspace();
  const [name, setName] = useState("");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<ProjectSort>("updated");
  const shown = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    const matching = needle ? ws.projects.filter((p) => p.name.toLocaleLowerCase().includes(needle)) : ws.projects;
    return [...matching].sort((a, b) => (sort === "name" ? a.name.localeCompare(b.name, undefined, { numeric: true }) : Number(b.updatedAt) - Number(a.updatedAt)));
  }, [ws.projects, query, sort]);
  const create = () => {
    const value = name.trim() || "Untitled";
    setName("");
    void ws.createProject(value);
  };
  const available = ws.capabilities ? ws.capabilities.providers.filter((p) => p.available).length : 0;

  return (
    <div className="flex min-h-screen flex-col bg-surface">
      <header className="flex h-12 items-center gap-2 px-4 shadow-[inset_0_-1px_0_var(--border-soft)]" data-titlebar>
        <TitleBarInset side="left" />
        <div className="flex items-center gap-2 font-semibold">
          <img className="size-[22px]" src="/icon.png" alt="" width={22} height={22} />
          <span>Urakatsu</span>
        </div>
        <span className="flex-1" />
        {ws.capabilities?.device && (
          <Badge className="min-w-0 shrink" title={ws.capabilities.device.description}>
            <span className="truncate">{ws.capabilities.device.description}</span>
          </Badge>
        )}
        <a className={buttonVariants({ variant: "ghost", size: "sm" })} href="/api/docs" target="_blank" rel="noreferrer">
          API docs
        </a>
        <Button variant="ghost" size="sm" onClick={() => ws.setModelsOpen(true)}>
          <Boxes size={14} /> Models
        </Button>
        <Button variant="ghost" size="sm" onClick={() => ws.setSettingsOpen(true)}>
          <Settings size={14} /> Settings
        </Button>
        <ThemeSwitch value={ws.theme} onChange={ws.setTheme} />
        <TitleBarInset side="right" />
      </header>

      <main className="mx-auto w-full max-w-[1120px] px-5 py-12">
        <h1 className="text-[28px] leading-9 font-semibold tracking-[-0.01em]">Projects</h1>
        <p className="mt-2 max-w-[60ch] text-base text-fg-2">
          Each project holds chapters of manga pages. Urakatsu detects, reads, translates and cleans every page, then typesets the
          translation back in.
        </p>
        <form
          className="mt-6 flex max-w-[620px] gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            create();
          }}
        >
          <Input
            className="h-9 text-[16px]"
            aria-label="Project name"
            placeholder="Project name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <Button type="submit" className="h-9">
            <Plus size={16} /> Create project
          </Button>
          <Button variant="outline" type="button" className="h-9" onClick={() => ws.setSourcesOpen(true)}>
            <Link2 size={16} /> From a source
          </Button>
        </form>
        {ws.capabilities && (
          <p className="mt-2 text-xs tracking-[0.005em] text-fg-3">
            {ws.capabilities.platform}. {available} of {ws.capabilities.providers.length} providers available.
          </p>
        )}

        <section className="mt-12">
          {!ws.ready ? null : ws.projects.length === 0 ? (
            <div className={EMPTY_BLOCK}>
              <p>No projects yet.</p>
              <p className="text-fg-2">Projects you create appear here.</p>
            </div>
          ) : (
            <>
              <div className="mb-4 flex items-center gap-2">
                <label className="relative block w-[280px] max-w-full max-[640px]:w-auto max-[640px]:flex-1">
                  <Search size={14} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-icon-2" />
                  <Input
                    className="w-full pl-[30px]"
                    type="search"
                    aria-label="Search projects"
                    placeholder="Search projects"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === "Escape" && setQuery("")}
                  />
                </label>
                <Select size="sm" label="Sort projects" value={sort} options={SORTS} onChange={setSort} className="w-[180px] shrink-0" />
                <span className="ml-auto text-xs tracking-[0.005em] whitespace-nowrap text-fg-2 tabular-nums max-[640px]:hidden">
                  {query.trim() ? `${shown.length} of ${plural(ws.projects.length, "project")}` : plural(ws.projects.length, "project")}
                </span>
              </div>
              {shown.length === 0 ? (
                <div className={EMPTY_BLOCK}>
                  <p>No projects match &ldquo;{query.trim()}&rdquo;.</p>
                  <Button variant="outline" size="sm" onClick={() => setQuery("")}>
                    <X size={14} /> Clear search
                  </Button>
                </div>
              ) : (
                <ul
                  className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-x-4 gap-y-5 max-[640px]:grid-cols-[repeat(auto-fill,minmax(128px,1fr))]"
                  aria-label="Projects"
                >
                  {shown.map((project) => (
                    <li key={project.id} className="group relative min-w-0">
                      <button
                        className="-m-1 box-content flex w-full cursor-pointer flex-col gap-0.5 rounded-card bg-transparent p-1 text-left text-fg transition-colors duration-[20ms] ease-in outline-none hover:bg-hover focus-visible:shadow-focus"
                        onClick={() => void ws.openProject(project.id)}
                        title={project.name}
                      >
                        <ProjectCover id={project.id} version={project.updatedAt} name={project.name} />
                        <span className="line-clamp-2 font-medium [overflow-wrap:anywhere]">{project.name}</span>
                        <span className="truncate text-xs tracking-[0.005em] text-fg-2 tabular-nums">
                          {plural(project.chapters, "chapter")}, {plural(project.pages, "page")}
                        </span>
                        <span className="truncate text-xs tracking-[0.005em] text-fg-3" title={formatTimestamp(project.updatedAt)}>
                          Updated {formatDay(Number(project.updatedAt))}
                        </span>
                      </button>
                      <Button
                        variant="outline"
                        size="icon-sm"
                        className="absolute top-2 right-2 opacity-0 shadow-raised transition-opacity duration-100 ease-out group-hover:opacity-100 hover:bg-danger-weak hover:text-danger focus-visible:opacity-100 active:bg-danger-weak [@media(hover:none)]:opacity-100"
                        title="Delete project"
                        aria-label={`Delete ${project.name}`}
                        onClick={() => {
                          if (window.confirm(`Delete ${project.name}? This removes its chapters, pages and renders.`)) void ws.deleteProject(project.id);
                        }}
                      >
                        <Trash2 size={14} />
                      </Button>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}

/** The linked series' cover, else the first page; the name's initial when the project has neither. */
export function ProjectCover({ id, version, name }: { id: string; version: number | bigint; name: string }) {
  const [broken, setBroken] = useState(false);
  if (broken) {
    return (
      <span className={cn("flex items-center justify-center text-[28px] leading-5 font-semibold text-fg-3", COVER)} aria-hidden="true">
        {name.trim().charAt(0).toUpperCase() || "?"}
      </span>
    );
  }
  return <img className={cn("block", COVER)} src={sources.projectCoverUrl(id, String(version))} alt="" loading="lazy" onError={() => setBroken(true)} />;
}

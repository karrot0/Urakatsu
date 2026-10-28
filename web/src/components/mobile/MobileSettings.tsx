// Settings on a phone: a grouped list of sections, each pushing its own page. The pages are the
// desktop sections (SettingsPage.tsx), restyled for the phone with `max-md:` there.

import { useEffect, useState } from "react";

import { useWorkspace } from "../../state/workspace";
import { SETTINGS_TABS, SettingsProvider, SettingsSection, useSettingsDraft, type SettingsTab } from "../SettingsPage";
import { ListIcon, ListRow, ListSection, Screen, type Enter } from "./glass";

// Sections in groups, as the phone's own Settings app groups them.
const GROUPS: SettingsTab[][] = [["appearance"], ["pipeline", "providers", "translation", "typesetting"], ["runtimes", "sources", "shortcuts"], ["about"]];

const isTab = (value: string | undefined): value is SettingsTab => SETTINGS_TABS.some(([id]) => id === value);

export function MobileSettings() {
  const ws = useWorkspace();
  const [section, setSection] = useState<SettingsTab | undefined>(isTab(ws.settingsTab) ? ws.settingsTab : undefined);
  const [enter, setEnter] = useState<Enter>("none");
  const { draft, update, status } = useSettingsDraft();

  // Opened for one section (Manage sources, Runtimes): go straight there.
  useEffect(() => {
    if (isTab(ws.settingsTab)) setSection(ws.settingsTab);
  }, [ws.settingsTab]);

  const saving = status === "saving" ? "Saving" : status === "saved" ? "All changes saved" : undefined;
  const tab = SETTINGS_TABS.find(([id]) => id === section);

  return (
    <SettingsProvider>
      {tab ? (
        <Screen
          title={tab[1]}
          large={false}
          enter={enter}
          back={{
            label: "Settings",
            onBack: (swiped) => {
              setEnter(swiped ? "none" : "pop");
              setSection(undefined);
            },
          }}
        >
          <div className="pt-3">
            <SettingsSection tab={tab[0]} draft={draft} update={update} />
          </div>
          {saving && (
            <p className="pt-4 text-center text-[13px] text-fg-3" role="status">
              {saving}
            </p>
          )}
        </Screen>
      ) : (
        <Screen title="Settings" enter={enter}>
          {GROUPS.map((group) => (
            <ListSection key={group[0]}>
              {group.map((id) => {
                const [, label, Icon] = SETTINGS_TABS.find(([tabId]) => tabId === id)!;
                return (
                  <ListRow
                    key={id}
                    leading={<ListIcon icon={Icon} />}
                    title={label}
                    chevron
                    onClick={() => {
                      setEnter("push");
                      setSection(id);
                    }}
                  />
                );
              })}
            </ListSection>
          ))}
          {ws.capabilities && (
            <p className="px-4 pt-6 text-center text-[13px] leading-[18px] text-fg-3">
              {ws.capabilities.device?.description ? `${ws.capabilities.device.description}. ` : ""}
              {ws.capabilities.platform}.
            </p>
          )}
        </Screen>
      )}
    </SettingsProvider>
  );
}

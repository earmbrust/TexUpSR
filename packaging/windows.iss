#ifndef AppVersion
#define AppVersion "0.1.0"
#endif
#ifndef SourceDir
#error SourceDir must be provided by the build workflow
#endif

[Setup]
AppId=TexUpSR
AppName=TexUpSR
AppVersion={#AppVersion}
AppPublisher=Elden Armbrust
DefaultDirName={autopf}\TexUpSR
DisableProgramGroupPage=yes
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=TexUpSR-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
ChangesEnvironment=yes
UninstallDisplayIcon={app}\trt_sr.exe

[Files]
Source: "{#SourceDir}\trt_sr.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Code]
function EnvironmentRoot(): Integer;
begin
  if IsAdminInstallMode then
    Result := HKLM
  else
    Result := HKCU;
end;

function EnvironmentKey(): String;
begin
  if IsAdminInstallMode then
    Result := 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment'
  else
    Result := 'Environment';
end;

procedure AddToPath();
var
  CurrentPath, Entry: String;
begin
  Entry := ExpandConstant('{app}');
  if not RegQueryStringValue(EnvironmentRoot(), EnvironmentKey(), 'Path', CurrentPath) then
    CurrentPath := '';
  if Pos(';' + Uppercase(Entry) + ';', ';' + Uppercase(CurrentPath) + ';') <> 0 then
    Exit;
  if (CurrentPath <> '') and (CurrentPath[Length(CurrentPath)] <> ';') then
    CurrentPath := CurrentPath + ';';
  CurrentPath := CurrentPath + Entry;
  if not RegWriteExpandStringValue(EnvironmentRoot(), EnvironmentKey(), 'Path', CurrentPath) then
    RaiseException('Could not update PATH');
  if not RegWriteStringValue(EnvironmentRoot(), 'Software\TexUpSR', 'InstallerPath', Entry) then
    RaiseException('Could not save PATH registration');
end;

procedure RemoveFromPath();
var
  RegisteredPath, CurrentPath, Remaining, Part: String;
  Sep: Integer;
begin
  if not RegQueryStringValue(EnvironmentRoot(), 'Software\TexUpSR', 'InstallerPath', RegisteredPath) then
    Exit;
  if not RegQueryStringValue(EnvironmentRoot(), EnvironmentKey(), 'Path', CurrentPath) then
    Exit;
  Remaining := '';
  while CurrentPath <> '' do
  begin
    Sep := Pos(';', CurrentPath);
    if Sep = 0 then
    begin
      Part := CurrentPath;
      CurrentPath := '';
    end
    else
    begin
      Part := Copy(CurrentPath, 1, Sep - 1);
      Delete(CurrentPath, 1, Sep);
    end;
    if CompareText(Trim(Part), Trim(RegisteredPath)) <> 0 then
    begin
      if Remaining <> '' then Remaining := Remaining + ';';
      Remaining := Remaining + Part;
    end;
  end;
  if not RegWriteExpandStringValue(EnvironmentRoot(), EnvironmentKey(), 'Path', Remaining) then
    RaiseException('Could not restore PATH');
  RegDeleteValue(EnvironmentRoot(), 'Software\TexUpSR', 'InstallerPath');
  RegDeleteKeyIfEmpty(EnvironmentRoot(), 'Software\TexUpSR');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then AddToPath();
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then RemoveFromPath();
end;
